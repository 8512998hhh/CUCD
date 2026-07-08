import sys
import os
import timeit

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import evaluation, experiment_manager, parsers, helpers
from utils import datasets_points as datasets
from model import model
from utils.experiment_manager import CfgNode


# =====================================================================
# 时间步掩码生成
# =====================================================================
def generate_temporal_mask(B: int, T: int, mask_ratio: float, device) -> torch.Tensor:
    """
    为 batch 内每个样本独立随机选取 mask_ratio 比例的时间步。
    返回 (B, T) bool 张量，True = 该时间步被掩码（需要重建）。
    """
    num_masked = max(1, int(T * mask_ratio))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        indices = torch.randperm(T, device=device)[:num_masked]
        mask[b, indices] = True
    return mask


# =====================================================================
# 重建损失（归一化 MSE，参考 MAE 设计）
# =====================================================================
def mask_recon_loss(pred_pixels: torch.Tensor, target_pixels: torch.Tensor) -> torch.Tensor:
    """
    在被掩码时间步上计算归一化 MSE 损失。
    pred_pixels / target_pixels: (N_masked, C_in, H, W)

    归一化方式：对每个 patch（每帧每个样本）在通道×空间维度上做 z-score，
    与原始 MAE 的归一化 pixel target 一致，避免低频分量主导损失。
    """
    N, C, H, W = pred_pixels.shape

    # 对 target 做 patch 级 z-score 归一化（沿 C*H*W 维度）
    target_flat = target_pixels.reshape(N, -1)           # (N, C*H*W)
    mean = target_flat.mean(dim=1, keepdim=True)
    std  = target_flat.std(dim=1, keepdim=True).clamp(min=1e-6)
    target_norm = ((target_flat - mean) / std).reshape(N, C, H, W)

    pred_flat = pred_pixels.reshape(N, -1)
    pred_norm = ((pred_flat - mean) / std).reshape(N, C, H, W)

    return F.mse_loss(pred_norm, target_norm)


# =====================================================================
# 主训练函数
# =====================================================================
def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        if cfg.TRAINER.EPOCHS <= 1:
            return 1.0
        return 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

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

    epochs         = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)
    T              = cfg.DATALOADER.TIMESERIES_LENGTH

    global_step = epoch_float = 0
    best_ssl_loss = float('inf')

    # 掩码比例：遮住 40% 的时间步（可在 yaml 中用 MASK_RATIO 覆盖）
    mask_ratio = getattr(cfg.TRAINER, 'MASK_RATIO', 0.4)

    print(f'Mask ratio: {mask_ratio:.0%}  ({max(1, int(T * mask_ratio))}/{T} frames masked per sample)')

    for epoch in range(1, epochs + 1):
        print(f'Starting Mask-Recon Pre-training Epoch {epoch}/{epochs}.')
        wandb.log({
            'lr':    scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        net.train()
        start    = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x          = batch['x'].to(device)          # (B, T, C_in, H, W)
            timestamps = batch['timestamps'].to(device)
            B_cur      = x.shape[0]

            # =============================================================
            # Step 1：生成随机时间步掩码
            # =============================================================
            mask = generate_temporal_mask(B_cur, T, mask_ratio, device)
            # mask: (B, T) bool，True = 掩码时间步

            # =============================================================
            # Step 2：特征级掩码重建前向传播（单次，梯度贯穿全程）
            #   - Encoder 对所有帧正常编码
            #   - Encoder 输出的特征图上将掩码帧置零
            #   - Mamba/TFR 被迫从可见帧补全掩码帧
            #   - 重建头将掩码帧特征映射回像素空间
            # =============================================================
            pred_pixels, target_pixels = model.masked_recon_forward(
                net, x, mask, edges, timestamps=timestamps
            )
            # pred_pixels   : (N_masked, C_in, H, W)
            # target_pixels : (N_masked, C_in, H, W)

            # =============================================================
            # Step 3：仅在掩码位置计算归一化 MSE 重建损失
            # =============================================================
            loss = mask_recon_loss(pred_pixels, target_pixels)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            global_step  += 1
            epoch_float   = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                mean_loss = np.mean(loss_set)
                n_masked  = mask.sum(dim=1).float().mean().item()
                print(
                    f'Step {global_step} (epoch {epoch_float:.2f}) | '
                    f'Recon Loss: {mean_loss:.4f} | '
                    f'Avg masked frames: {n_masked:.1f}/{T}'
                )
                wandb.log({
                    'loss_mask_recon': mean_loss,
                    'time':            timeit.default_timer() - start,
                    'step':            global_step,
                    'epoch':           epoch_float,
                })
                start    = timeit.default_timer()
                loss_set = []

        # ================== Epoch 结束 ==================
        if scheduler is not None:
            scheduler.step()

        # epoch 末尾可能 loss_set 已被清空，用 0 占位
        current_epoch_loss = np.mean(loss_set) if len(loss_set) > 0 else 0.0

        # 每个 epoch 结束都覆盖保存最新权重
        model.save_model_latest(net, epoch, cfg)
        print(f'Saved latest network (Epoch {epoch})', flush=True)

        # loss 下降时额外保存一份 best 权重
        if 0 < current_epoch_loss < best_ssl_loss:
            best_ssl_loss = current_epoch_loss
            wandb.log({'best_ssl_loss': best_ssl_loss, 'epoch': epoch_float})
            print(f'>>> New Best SSL Loss: {best_ssl_loss:.4f}, saving best network <<<', flush=True)
            model.save_model_pretrained(net, epoch, cfg)



if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg  = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=== Running Mask-Recon Pre-training on device:', device)

    wandb.init(
        name=cfg.NAME + '_MaskRecon_Pretrain',
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
