import sys
import os
import timeit

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np
from tqdm import tqdm

from utils import datasets_points, evaluation, experiment_manager, parsers, helpers
from model import model
from utils.helpers import active_learning_sampling, generate_active_labels_from_points
from utils.experiment_manager import CfgNode
from sample import run_or_load_active_sampling


# ==============================================================================

def slice_to_patches(tensor, patch_size=256):
    """
    无缝将 (B, T, C, H, W) 的大图切分为 (B * Num_Patches, T, C, patch_size, patch_size)
    适用于 H 和 W 能被 patch_size 整除的情况 (例如 768 和 1024)
    """
    B, T, C, H, W = tensor.shape
    num_y = H // patch_size
    num_x = W // patch_size
    
    # 1. 重塑：将 H 和 W 拆分成 (网格数, 网格大小)
    reshaped = tensor.view(B, T, C, num_y, patch_size, num_x, patch_size)
    
    # 2. 置换维度：把所有的空间网格维度提到前面去
    permuted = reshaped.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    
    # 3. 展平批次：把 (B, num_y, num_x) 合并成一个新的大的 Batch
    patches = permuted.view(-1, T, C, patch_size, patch_size)
    
    return patches

def run_training(cfg: CfgNode):
    net = model.load_model_IL(cfg, device)
    
    net.to(device)

    # 🌟 [关键点 2]：微调学习率应该比预训练小很多 (比如 1e-4 或 1e-5)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion_point = model.point_power_jaccard_loss
    # 删除了 criterion_edge

    # 🌟 [关键点 3]：关闭空间增强，防止采样点与图像错位！
    start_time = cfg.DATASET.START_TIME
    end_time = cfg.DATASET.END_TIME
    dataset = datasets_points.TrainSpaceNet7ActiveDataset(cfg, run_type="train", no_augmentations=True, disable_multiplier = True, start_time=start_time, end_time=end_time)
    print(dataset)

    # 为了方便按顺序采样和映射，采样时的 DataLoader 不要 shuffle
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': False,  # 采样阶段不打乱
        'drop_last': False, # 保证所有图都被采样
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # edges_pre = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, 'adjacent')
    active_labels_memory = run_or_load_active_sampling(cfg, net, dataloader, device, num_points=5)
    
    # 解锁 Shuffle，准备进入微调训练阶段
    dataset_train = datasets_points.create_train_dataset(cfg, run_type='train', no_augmentations=False)
    dataloader_kwargs['shuffle'] = cfg.DATALOADER.SHUFFLE
    dataloader_kwargs['drop_last'] = True
    dataloader_train = torch_data.DataLoader(dataset_train, **dataloader_kwargs)

    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader_train)
    global_step = epoch_float = 0
    best_f1_val = 0

    # ========================================================================
    # 🚀 阶段二：使用固定采样点进行增量微调 (Incremental Fine-tuning Phase)
    # ========================================================================
    print("\n========== Phase 2: 开始增量微调 (Fine-tuning) ==========")
    # dataloader_train 的 batch_size 依然设为 1
    mini_batch_size = 24 # 显卡吃力就设为 1，显卡好就设为 4

    for epoch in range(1, epochs + 1):
        net.train()
        loss_seg_set, loss_ch_set, loss_set = [], [], []
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})
        start = timeit.default_timer()
        # 加上 tqdm 进度条更直观
        for i, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch}/{epochs}"):
            # 注意：不要在小切片循环里 zero_grad，要在一张大图开始前 zero_grad
            optimizer.zero_grad()

            x_large = batch['x'].to(device)
            aoi = batch['aoi_id'][0]
            
            # 🚀 1. 【核心新增】：动态获取当前 Batch 真实的序列长度 T
            current_T = x_large.size(1) 
            if current_T < 2:
                # 理论上 DataLoader 已经过滤了，这里作为双重保险
                continue 
                
            # 🚀 2. 【核心新增】：动态生成属于这个长度的 edges
            current_edges = helpers.get_edges(current_T, 'adjacent')
            
            # 取出 Phase 1 算好的切片标签 
            y_point_patches = active_labels_memory[aoi]['y_point'].to(device)
            loss_mask_patches = active_labels_memory[aoi]['loss_mask'].to(device)
            
            # 将当前的大图切成小块
            x_patches = slice_to_patches(x_large, patch_size=64) # 注意这里改成了 64
        
            
            # 如果包含变化标签，同步切片
            if 'y_ch' in batch:
                y_ch_large = batch['y_ch'].to(device)
                if y_ch_large.dim() == 4:
                    y_ch_large = y_ch_large.unsqueeze(2)
                y_ch_patches = slice_to_patches(y_ch_large, patch_size=64)
                
            num_patches = x_patches.size(0) 
            
            # 用于记录整张大图的真实 Loss
            total_loss_seg_record = 0.0
            total_loss_ch_record = 0.0
            total_loss_record = 0.0
            loss_seg_set = []
            
            # 【微批次梯度累加】
            for start_idx in range(0, num_patches, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, num_patches)
                
                mb_x = x_patches[start_idx:end_idx]
                mb_y_point = y_point_patches[start_idx:end_idx]
                mb_loss_mask = loss_mask_patches[start_idx:end_idx]
                
                
                # 🚀 5. 【核心新增】：前向传播传入动态 current_edges 和 mb_ts
                logits_ch, logits_seg = net(mb_x, current_edges)
                
                loss_seg = criterion_point(logits_seg, mb_y_point, mb_loss_mask)
                
                if 'y_ch' in batch:
                    mb_y_ch = y_ch_patches[start_idx:end_idx]
                    mb_y_ch_point = mb_y_ch * mb_loss_mask.unsqueeze(1)
                else:
                    # 🚀 6. 【核心新增】：这里的 target 生成也要用 current_edges
                    mb_y_ch_point = helpers.get_ch(mb_y_point, current_edges)
                    
                loss_ch = criterion_point(logits_ch, mb_y_ch_point, mb_loss_mask)
                
                # 损失归一化
                loss = (loss_seg + loss_ch) / (num_patches / mini_batch_size)
                loss.backward()
                
                # 修复 Bug：累加真实的非归一化 loss 以便记录
                total_loss_seg_record += loss_seg.item() * (end_idx - start_idx) / num_patches
                total_loss_ch_record += loss_ch.item() * (end_idx - start_idx) / num_patches
                total_loss_record += loss.item() * (num_patches / mini_batch_size) # 还原为整图 loss
                
            # 12/192 个切片全部过完后，再统一执行一次权重更新！
            optimizer.step()

            # 🚀 使用累加的整图真实 loss 记录
            loss_seg_set.append(total_loss_seg_record)
            loss_ch_set.append(total_loss_ch_record)
            loss_set.append(total_loss_record)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                time_elapsed = timeit.default_timer() - start
                wandb.log({
                    'loss_seg': np.mean(loss_seg_set),
                    'loss_ch': np.mean(loss_ch_set),
                    'loss': np.mean(loss_set),
                    'time': time_elapsed,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_set = [], [], []

        # ======== 验证与保存阶段 ========
        if scheduler is not None:
            scheduler.step()
            
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val >= best_f1_val:
            best_f1_val = f1_val
            wandb.log({
                'best val f1': best_f1_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            model.save_model(net, epoch, cfg)
        model.save_model_latest(net, epoch, cfg)

    # 测试最优模型
    net = model.load_model(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)

if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_manager.save_experiment_config(cfg)
    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project='ContUrbanCD',
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)