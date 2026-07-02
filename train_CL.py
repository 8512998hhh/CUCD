import sys
import os
import timeit

import torch
import torch.nn.functional as F  # 用于计算余弦相似度和交叉熵
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import evaluation, experiment_manager, parsers, helpers
from utils import datasets_points as datasets
from model import model
from utils.experiment_manager import CfgNode

# =====================================================================
# 🚀 新增：空间低频互换 (抗季节/光照干扰)
# =====================================================================
def spatial_low_freq_swap(x, beta=0.1):
    """
    对图像的空间维度(H, W)进行 2D-FFT，互换中心低频振幅。
    beta 控制互换的低频窗口大小比例。
    """
    B, T, C, H, W = x.shape
    
    # 1. 空间 2D FFT (将低频移到中心)
    fft_x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    amp = torch.abs(fft_x)
    phase = torch.angle(fft_x)
    
    # 2. 生成 Batch 内的随机打乱索引 (用来提供替换的背景)
    idx = torch.randperm(B, device=x.device)
    amp_shuffled = amp[idx]
    
    # 3. 制作低频掩码 (Mask)
    mask = torch.zeros_like(amp)
    h_center, w_center = H // 2, W // 2
    h_window, w_window = int(H * beta), int(W * beta)
    
    # 防止窗口大小为 0
    h_window = max(1, h_window)
    w_window = max(1, w_window)
    
    mask[..., h_center - h_window : h_center + h_window, 
              w_center - w_window : w_center + w_window] = 1.0
    
    # 4. 混合振幅: 中心低频用别人的，边缘高频用自己的
    amp_mixed = mask * amp_shuffled + (1.0 - mask) * amp
    
    # 5. 逆 FFT 重构图像
    fft_mixed = amp_mixed * torch.exp(1j * phase)
    fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    x_aug = torch.fft.ifft2(fft_mixed_unshifted, dim=(-2, -1)).real
    
    return x_aug

# =====================================================================
# 🚀 新增：时间高频互换 (抗云层/瞬态噪声干扰)
# =====================================================================
def temporal_high_freq_swap(x, alpha=0.3):
    """
    对图像的时间维度(T)进行 1D-FFT，互换高频振幅。
    alpha 控制互换的高频截断比例。
    """
    B, T, C, H, W = x.shape
    
    # 1. 时间 1D FFT (针对实数的快速 FFT)
    fft_x = torch.fft.rfft(x, dim=1)
    amp = torch.abs(fft_x)
    phase = torch.angle(fft_x)
    
    # 2. 生成 Batch 内的打乱索引
    idx = torch.randperm(B, device=x.device)
    amp_shuffled = amp[idx]
    
    # 3. 制作高频掩码 (高频在 rfft 结果的尾部)
    freq_bins = amp.shape[1]
    mask = torch.zeros_like(amp)
    cutoff = int(freq_bins * (1.0 - alpha)) 
    mask[:, cutoff:, ...] = 1.0
    
    # 4. 混合振幅: 尾部高频用别人的，头部低频用自己的
    amp_mixed = mask * amp_shuffled + (1.0 - mask) * amp
    
    # 5. 逆 FFT 重构图像序列
    fft_mixed = amp_mixed * torch.exp(1j * phase)
    x_aug = torch.fft.irfft(fft_mixed, n=T, dim=1)
    
    return x_aug
# =====================================================================

def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion = model.power_jaccard_loss
    # reset the generators
    dataset = datasets.create_train_dataset(cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.MODEL.EDGE_TYPE)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val = 0
    trigger_times = 0
    stop_training = False
    
    # 🚀 修改：频域对比学习超参数设置
    lambda_contrastive = getattr(cfg.TRAINER, 'LAMBDA_CONTRASTIVE', 0.1)
    beta_spatial = getattr(cfg.TRAINER, 'BETA_SPATIAL', 0.1)   # 空间低频互换比例
    alpha_temporal = getattr(cfg.TRAINER, 'ALPHA_TEMPORAL', 0.3) # 时间高频互换比例

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})
        start = timeit.default_timer()
        
        # 追踪列表
        loss_seg_set, loss_ch_set, loss_c_set, loss_set = [], [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x, y_seg = batch['x'].to(device), batch['y'].to(device)
            timestamps = batch['timestamps'].to(device)
            
            # =================================================================
            # 🚀 核心改动：一师两徒 (Teacher + 2 Students) 频域多视图前向传播
            # =================================================================
            
            # 1. 完整视图 (Teacher Anchor) - 计算主要任务 Loss 并提取锚点特征
            logits_ch, logits_seg, features_full = net(x, edges, timestamps=timestamps, return_features=True)
            y_ch = helpers.get_ch(y_seg, edges)
            
            # 为了防止特征崩塌并节省显存，Teacher 的特征使用 detach()，不传导梯度
            anchor_feat = features_full.detach()
            
            # 2. 学生视图 A：空间低频互换 (抗季节/光照)
            x_spatial = spatial_low_freq_swap(x.clone(), beta=beta_spatial)
            _, _, features_spatial = net(x_spatial, edges, timestamps=timestamps, return_features=True)
            
            # 3. 学生视图 B：时间高频互换 (抗云层/瞬态噪声)
            x_temporal = temporal_high_freq_swap(x.clone(), alpha=alpha_temporal)
            _, _, features_temporal = net(x_temporal, edges, timestamps=timestamps, return_features=True)
            
            # =================================================================

            # 计算主要任务 Loss
            # loss_seg = criterion(logits_seg, y_seg)
            # loss_ch = criterion(logits_ch, y_ch)
            
            # 🚀 计算多视图 InfoNCE 对比损失 (采样 2048 个点)
            # 强制空间互换特征与原特征对齐
            loss_c_spatial = model.info_nce_loss_pixel_sample(anchor_feat, features_spatial, temperature=0.1, num_samples=2048)
            # 强制时间互换特征与原特征对齐
            loss_c_temporal = model.info_nce_loss_pixel_sample(anchor_feat, features_temporal, temperature=0.1, num_samples=2048)
            
            # 对比损失求平均
            loss = loss_c_spatial + loss_c_temporal

            # 融合总 Loss
            #loss = loss_seg + loss_ch + lambda_contrastive * loss_c
            
            loss.backward()
            optimizer.step()

            # 记录 Loss
            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_c_set.append(loss_c.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss_seg': np.mean(loss_seg_set),
                    'loss_ch': np.mean(loss_ch_set),
                    'loss_contrastive': np.mean(loss_c_set), 
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_c_set, loss_set = [], [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        
        if scheduler is not None:
            scheduler.step()
            
        # evaluation at the end of an epoch
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

    # 测试阶段保持不变
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