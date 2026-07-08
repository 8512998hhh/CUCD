import sys
import os
import timeit

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm

from utils import datasets_points, evaluation, experiment_manager, parsers, helpers
from model import model
from utils.helpers import active_learning_sampling, generate_active_labels_from_points
from utils.experiment_manager import CfgNode

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

def run_or_load_active_sampling(cfg, net, dataloader, device, patch_size=64, num_points=1, nms_radius=5):
    """
    执行主动学习采样，或者从本地缓存加载已有的采样点。
    """
    save_path = Path(cfg.PATHS.OUTPUT) / 'samples' / 'active_sample.pt'
    # 🌟 1. 智能拦截：如果缓存文件已存在，直接 1 秒钟加载完毕并返回！
    if os.path.exists(save_path):
        print(f"\n✅ 发现已缓存的采样点文件: {save_path}")
        print("⚡ 正在直接加载本地缓存，跳过繁重的模型推理采样阶段...")
        active_labels_memory = torch.load(save_path)
        print(f"加载成功！共包含 {len(active_labels_memory)} 个区域的标注点。")
        return active_labels_memory

    # 🌟 2. 如果没有缓存，则老老实实跑 Phase 1 采样流程
    print(f"\n========== Phase 1: 未发现缓存，正在进行主动学习采样 ==========")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    active_labels_memory = {}
    net.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Phase 1: 采样推理中"):
            x_large = batch['x'].to(device)
            gt_large = batch['y'].to(device)
            current_T = x_large.size(1) 
            
            # 这里调用你的 edges 函数
            edges_pre = helpers.get_edges(current_T, 'adjacent')
            
            # 切片
            x_patches = slice_to_patches(x_large, patch_size=patch_size)
            gt_patches = slice_to_patches(gt_large, patch_size=patch_size)
            
            # 前向推理
            logits_ch, logits_seg = net(x_patches, edges_pre)
            
            # 采点
            sampled_points, _ = active_learning_sampling(
                logits_seg, logits_ch, 
                num_points=num_points,   
                nms_radius=nms_radius
            )
            
            # 生成 Mask
            y_point_patches, mask_patches = generate_active_labels_from_points(sampled_points, gt_patches)
            
            # 存入字典
            aoi = batch['aoi_id'][0]
            active_labels_memory[aoi] = {
                'y_point': y_point_patches.cpu(),
                'loss_mask': mask_patches.cpu()
            }

    print(f"采样完成！共计算了 {len(active_labels_memory)} 个区域。")
    
    # 🌟 3. 将算好的字典永久保存到硬盘
    # 自动创建所需的文件夹
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(active_labels_memory, save_path)
    print(f"💾 采样结果已永久保存至: {save_path}！")
    
    return active_labels_memory

if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.load_model_IL(cfg, device)
    net.to(device)
    dataset = datasets_points.create_train_dataset(cfg, run_type='train', no_augmentations=True)
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
    print('=== Runnning on device: p', device)
    run_or_load_active_sampling(cfg, net, dataloader, device, patch_size=64, num_points=1, nms_radius=5)