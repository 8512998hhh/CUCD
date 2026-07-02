# =====================================================================
# train_SSL_CLv3.py
#
# 主要方法：
#   1. L_dist：相邻帧不再做逐像素拉近，而是在独立投影子空间中做
#      feature distribution alignment（symmetric KL），减少成像条件域偏移，
#      同时避免直接抹平真实变化区域。
#   2. L_inst：在另一个独立投影子空间中做传统 instance contrastive learning。
#      每一帧图像与其颜色增强视图为正样本，batch 内所有其它帧图像
#      （包括同一序列的其它时间帧）都作为负样本。
#   3. L_cc：保留变化一致性约束，在共享下游特征上约束原图序列和增强序列
#      的相邻帧变化强度图一致。
#
# 下游微调只加载模型主体权重；这里的投影头只是预训练辅助子空间，不保存。
# =====================================================================
import sys
import os
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import experiment_manager, parsers, helpers
from utils import datasets_points as datasets
from model import model
from utils.experiment_manager import CfgNode


class PixelProjectionHead(nn.Module):
    """逐像素 1x1 Conv 投影头，用作 SSL 训练时的临时子空间。"""
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = 128):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = features.shape
        x = features.reshape(B * T, C, H, W)
        x = self.proj(x)
        return x.reshape(B, T, x.size(1), H, W)


def symmetric_kl_temporal_distribution_loss(
    features: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    相邻帧特征分布对齐，而不是逐像素拉近。

    features: (B, T, C, H, W)
    """
    B, T, C, _, _ = features.shape
    if T < 2:
        return features.new_tensor(0.0)

    pooled = features.mean(dim=(-2, -1))  # (B, T, C)
    p = F.softmax(pooled[:, :-1] / temperature, dim=-1).clamp_min(eps)
    q = F.softmax(pooled[:, 1:] / temperature, dim=-1).clamp_min(eps)

    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def instance_contrastive_loss(
    features: torch.Tensor,
    features_aug: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    """
    传统 instance CL。

    每个样本单元是一帧图像 (b, t)。正样本是同一帧的增强视图；
    负样本是 batch 中所有其它帧图像，包括同一序列的其它时间帧。
    """
    B, T, C, _, _ = features.shape
    z = F.normalize(features.mean(dim=(-2, -1)).reshape(B * T, C), dim=-1)
    z_aug = F.normalize(features_aug.mean(dim=(-2, -1)).reshape(B * T, C), dim=-1)
    logits = torch.matmul(z, z_aug.T) / temperature
    labels = torch.arange(B * T, device=features.device)
    return F.cross_entropy(logits, labels)


def _normalize_change_map(change_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对每个样本的变化强度图做 z-score 归一化，避免不同 batch / 时间跨度的
    绝对差异尺度主导 LCC。

    change_map: (B, E, 1, H, W)
    """
    B = change_map.size(0)
    flat = change_map.reshape(B, -1)
    mean = flat.mean(dim=1).view(B, 1, 1, 1, 1)
    std = flat.std(dim=1).clamp(min=eps).view(B, 1, 1, 1, 1)
    return (change_map - mean) / std


def change_consistency_loss(
    features_ref: torch.Tensor,
    features_aug: torch.Tensor,
) -> torch.Tensor:
    """
    LCC: 约束原图序列与增强序列的变化响应一致。

    features_ref / features_aug: (B, T, C, H, W)，使用 project=False 的下游特征。
    这里不直接拉近单帧特征，而是拉近相邻帧差异图：
        ||F_t - F_{t-1}||_2  vs  ||F'_t - F'_{t-1}||_2
    这样鼓励变化结构对颜色/对比度扰动稳定，而不是抹平真实变化。
    """
    if features_ref.size(1) < 2:
        return features_aug.new_tensor(0.0)

    with torch.no_grad():
        diff_ref = features_ref[:, 1:] - features_ref[:, :-1]
        change_ref = torch.norm(diff_ref, p=2, dim=2, keepdim=True)
        change_ref = _normalize_change_map(change_ref)

    diff_aug = features_aug[:, 1:] - features_aug[:, :-1]
    change_aug = torch.norm(diff_aug, p=2, dim=2, keepdim=True)
    change_aug = _normalize_change_map(change_aug)

    return F.mse_loss(change_aug, change_ref)


def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    feature_dim = cfg.MODEL.TOPOLOGY[0]
    proj_dist = PixelProjectionHead(feature_dim, out_channels=128).to(device)
    proj_inst = PixelProjectionHead(feature_dim, out_channels=128).to(device)

    optimizer = optim.AdamW(
        list(net.parameters()) + list(proj_dist.parameters()) + list(proj_inst.parameters()),
        lr=cfg.TRAINER.LR,
        weight_decay=0.01,
    )

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

    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    global_step = epoch_float = 0
    best_ssl_loss = float('inf')

    dist_temperature = getattr(cfg.TRAINER, 'DIST_TEMPERATURE', 1.0)
    inst_temperature = getattr(cfg.TRAINER, 'INST_TEMPERATURE', getattr(cfg.TRAINER, 'TEMPERATURE', 0.15))
    lambda_dist = getattr(cfg.TRAINER, 'LAMBDA_DIST', 1.0)
    lambda_inst = getattr(cfg.TRAINER, 'LAMBDA_INST', 0.5)
    lambda_cc = getattr(cfg.TRAINER, 'LAMBDA_CC', 0.1)
    cc_brightness = getattr(cfg.TRAINER, 'CC_BRIGHTNESS', 0.2)
    cc_contrast = getattr(cfg.TRAINER, 'CC_CONTRAST', 0.2)

    print(
        f'CLv3: lambda_dist={lambda_dist}, lambda_inst={lambda_inst}, lambda_cc={lambda_cc}, '
        f'dist_temperature={dist_temperature}, inst_temperature={inst_temperature}'
    )

    for epoch in range(1, epochs + 1):
        print(f'Starting CLv3 Pre-training Epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        net.train()
        proj_dist.train()
        proj_inst.train()
        start = timeit.default_timer()
        loss_set, loss_dist_set, loss_inst_set, loss_cc_set = [], [], [], []
        epoch_loss_set = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            timestamps = batch['timestamps'].to(device)

            features = net(x, edges, timestamps=timestamps, return_features=True, project=False)

            # =================================================================
            # 1) L_dist：相邻帧特征分布对齐（symmetric KL），不做逐像素拉近
            # =================================================================
            loss_dist = None
            if lambda_dist > 0:
                features_dist = proj_dist(features)
                loss_dist = symmetric_kl_temporal_distribution_loss(
                    features_dist,
                    temperature=dist_temperature,
                )
                (loss_dist * lambda_dist).backward(retain_graph=(lambda_inst > 0))
                del features_dist

            # =================================================================
            # 2) 增强分支：L_inst 与 LCC 共享同一次增强图前向
            #    - L_inst 在 proj_inst 子空间做传统 CL
            #    - LCC 在共享下游特征空间做变化强度图一致性
            # =================================================================
            loss_inst = None
            loss_cc = None
            if lambda_inst > 0 or lambda_cc > 0:
                x_aug = model.color_augment_batch(
                    x,
                    brightness=cc_brightness,
                    contrast=cc_contrast,
                )
                features_aug = net(x_aug, edges, timestamps=timestamps, return_features=True, project=False)

                if lambda_inst > 0:
                    features_inst = proj_inst(features)
                    features_aug_inst = proj_inst(features_aug)
                    loss_inst = instance_contrastive_loss(
                        features_inst,
                        features_aug_inst,
                        temperature=inst_temperature,
                    )
                    (loss_inst * lambda_inst).backward(retain_graph=(lambda_cc > 0))
                    del features_inst, features_aug_inst

                if lambda_cc > 0:
                    loss_cc = change_consistency_loss(features.detach(), features_aug)
                    (loss_cc * lambda_cc).backward()

                del features_aug, x_aug
            del features

            optimizer.step()

            loss_dist_val = loss_dist.item() if loss_dist is not None else 0.0
            loss_inst_val = loss_inst.item() if loss_inst is not None else 0.0
            loss_cc_val = loss_cc.item() if loss_cc is not None else 0.0
            loss_val = lambda_dist * loss_dist_val + lambda_inst * loss_inst_val + lambda_cc * loss_cc_val
            loss_set.append(loss_val)
            epoch_loss_set.append(loss_val)
            loss_dist_set.append(loss_dist_val)
            loss_inst_set.append(loss_inst_val)
            loss_cc_set.append(loss_cc_val)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                mean_loss = np.mean(loss_set)
                print(
                    f'Logging step {global_step} (epoch {epoch_float:.2f}) | '
                    f'SSL Loss: {mean_loss:.4f} | '
                    f'L_dist: {np.mean(loss_dist_set):.4f} | '
                    f'L_inst: {np.mean(loss_inst_set):.4f} | '
                    f'L_cc: {np.mean(loss_cc_set):.4f}'
                )

                wandb.log({
                    'loss_ssl_clv3': mean_loss,
                    'loss_dist': np.mean(loss_dist_set),
                    'loss_inst': np.mean(loss_inst_set),
                    'loss_cc': np.mean(loss_cc_set),
                    'time': timeit.default_timer() - start,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, loss_dist_set, loss_inst_set, loss_cc_set = [], [], [], []

        if scheduler is not None:
            scheduler.step()

        current_epoch_loss = np.mean(epoch_loss_set) if len(epoch_loss_set) > 0 else 0

        model.save_model_latest(net, epoch, cfg)
        print(f'Saved latest network (Epoch {epoch})', flush=True)

        if 0 < current_epoch_loss < best_ssl_loss:
            best_ssl_loss = current_epoch_loss
            wandb.log({'best_ssl_loss': best_ssl_loss, 'epoch': epoch_float})
            print(f'>>> New Best SSL Loss: {best_ssl_loss:.4f}, saving best network <<<', flush=True)
            model.save_model_pretrained(net, epoch, cfg)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Running CLv3 SSL Pre-training on device:', device)

    wandb.init(
        name=cfg.NAME + "_SSL_CLv3",
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
