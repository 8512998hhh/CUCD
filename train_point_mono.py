import sys
import os
import timeit

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import datasets_points, evaluation, experiment_manager, parsers, helpers
from model import model
from utils.experiment_manager import CfgNode


# =====================================================================
# 点监督 + 时序平滑先验 联合训练
#
# 主任务：
#   - 点监督建筑分割 loss_seg
#   - 点监督变化检测 loss_ch
#
# 辅助任务 L_smooth（直接作用在 logits_seg，无投影头）：
#   数据先验：建筑在时间轴上状态翻转次数极少，且持久——不会在短时间内
#   来回横跳。这对应两个可微约束：
#
#   1) L_tv（一阶时序总变差）：
#      惩罚相邻帧预测概率的差值绝对值。
#      L1-TV 鼓励稀疏跳变（允许少数真实建造/拆除），只压噪声式小抖动。
#      用 Δt 加权：相邻帧时间间隔越短，平滑性惩罚越强（短期不该有大变化）。
#
#   2) L_curv（二阶差分，可选）：
#      惩罚"变了又变回去"的反转（0→1→0）。
#      一阶 TV 压所有变化；二阶差分专门压来回横跳——单次持久变化
#      的二阶几乎为零，只有反复横跳才被重罚。
#
# 为什么比对比损失有效：
#   - 稠密：每个像素每对相邻帧都有梯度，直接填补稀疏点监督的空白；
#   - 任务对齐：直接作用在 logits_seg，没有投影头的间接路径，无负迁移；
#   - 无标签：纯结构先验，不依赖任何标注；
#   - 契合数据：SN7 城市发展时序里，建筑状态确实近乎低频变化。
# =====================================================================


def temporal_smoothness_loss(
    logits_seg: torch.Tensor,
    delta_t: torch.Tensor,
    lambda_tv: float = 1.0,
    lambda_curv: float = 0.0,
    delta_t_eps: float = 1.0,
):
    """
    Δt 归一化的时序平滑损失，直接作用在 logits_seg 上。

    logits_seg: (B, T, 1, H, W)，分割头原始输出（未经 sigmoid）
    delta_t:    (B, T, 1)，相邻帧时间间隔（月数），第 0 帧为 0
                来自 batch['timestamps']，形状 (B, T, 1)
    lambda_tv:   一阶 TV 项权重
    lambda_curv: 二阶差分项权重（0 表示关闭）
    delta_t_eps: 防止除零的最小间隔值

    返回标量 loss。
    """
    prob = torch.sigmoid(logits_seg)          # (B, T, 1, H, W)
    B, T, _, H, W = prob.shape

    if T < 2:
        return prob.sum() * 0.0

    # delta_t: (B, T, 1) -> 取相邻帧的间隔，即 delta_t[:, 1:, :]
    # 第 i 帧的 delta_t 定义为"第 i 帧与第 i-1 帧的月数差"，第 0 帧为 0
    # 用于加权的间隔是相邻差分 (t, t+1) 的时间跨度
    dt = delta_t[:, 1:, :]                    # (B, T-1, 1)
    dt = dt.clamp(min=delta_t_eps)            # 防止 delta_t=0 导致权重爆炸

    # 时序权重：间隔越短 → 惩罚越强（短时间内不该有变化）
    # 用 1/dt 归一化，并做 softmax-like 归一化只是为了量级稳定
    # 简单版：w = 1 / dt，clip 防止过大
    w = (1.0 / dt).clamp(max=10.0)            # (B, T-1, 1)，广播到 H, W

    # ---- 一阶 TV ----
    diff1 = prob[:, 1:] - prob[:, :-1]        # (B, T-1, 1, H, W)
    # w: (B, T-1, 1) -> (B, T-1, 1, 1, 1) 广播
    w_spatial = w.unsqueeze(-1).unsqueeze(-1)  # (B, T-1, 1, 1, 1)
    loss_tv = (w_spatial * diff1.abs()).mean()

    if lambda_curv <= 0.0 or T < 3:
        return lambda_tv * loss_tv

    # ---- 二阶差分（惩罚来回横跳）----
    # d2[t] = diff1[t+1] - diff1[t]，即概率曲线的"加速度"
    diff2 = diff1[:, 1:] - diff1[:, :-1]      # (B, T-2, 1, H, W)

    # 二阶的时间权重：取相邻两段间隔的调和均值
    dt2 = (dt[:, :-1] + dt[:, 1:]) / 2.0      # (B, T-2, 1)
    w2 = (1.0 / dt2.clamp(min=delta_t_eps)).clamp(max=10.0)
    w2_spatial = w2.unsqueeze(-1).unsqueeze(-1)

    loss_curv = (w2_spatial * diff2.abs()).mean()

    return lambda_tv * loss_tv + lambda_curv * loss_curv


def run_training(cfg: CfgNode):
    if getattr(cfg.TRAINER, 'PRETRAINED', False):
        net = model.load_model(cfg, device, pretrained=True)
        print("Pretrained Model Loaded.")
    else:
        net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        if cfg.TRAINER.EPOCHS <= 1:
            return 1.0
        return 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion_point = model.point_power_jaccard_loss

    dataset = datasets_points.create_train_dataset(cfg, run_type='train')
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
    best_f1_val = 0

    # 时序平滑超参
    lambda_smooth = getattr(cfg.TRAINER, 'LAMBDA_SMOOTH', 0.05)
    lambda_tv     = getattr(cfg.TRAINER, 'LAMBDA_TV',     1.0)
    lambda_curv   = getattr(cfg.TRAINER, 'LAMBDA_CURV',   0.0)
    # 延迟启动：前 N epoch 辅助损失完全关闭，让主任务先站稳
    smooth_delay_epochs = getattr(cfg.TRAINER, 'SMOOTH_DELAY_EPOCHS', 10)

    print(
        f'Point + Smooth: lambda_smooth={lambda_smooth}, '
        f'lambda_tv={lambda_tv}, lambda_curv={lambda_curv}, '
        f'smooth_delay_epochs={smooth_delay_epochs}'
    )

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_sup_set = [], [], []
        loss_smooth_set, loss_set = [], []

        # 延迟启动：前 smooth_delay_epochs 个 epoch 辅助损失完全关闭
        smooth_scale = 1.0 if epoch > smooth_delay_epochs else 0.0

        for i, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            x          = batch['x'].to(device)
            y_point    = batch['y_point'].to(device)
            loss_mask  = batch['loss_mask'].to(device)
            timestamps = batch['timestamps'].to(device)    # (B, T, 1)

            # =============================================================
            # 1) 主任务前向 + 点监督损失
            # =============================================================
            logits_ch, logits_seg = net(x, edges, timestamps)
            loss_seg = criterion_point(logits_seg, y_point, loss_mask)

            y_ch_point = helpers.get_ch(y_point, edges)
            loss_ch    = criterion_point(logits_ch, y_ch_point, loss_mask)

            loss_sup = loss_seg + loss_ch

            # =============================================================
            # 2) 时序平滑先验（前 smooth_delay_epochs 个 epoch 完全关闭）
            # =============================================================
            if smooth_scale > 0:
                loss_smooth = temporal_smoothness_loss(
                    logits_seg,
                    timestamps,
                    lambda_tv=lambda_tv,
                    lambda_curv=lambda_curv,
                )
                loss = loss_sup + lambda_smooth * loss_smooth
            else:
                loss_smooth = loss_sup.new_tensor(0.0)
                loss = loss_sup
            loss.backward()
            optimizer.step()

            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_sup_set.append(loss_sup.item())
            loss_smooth_set.append(loss_smooth.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(
                    f'Step {global_step} (epoch {epoch_float:.2f}) | '
                    f'Loss: {np.mean(loss_set):.4f} | '
                    f'Sup: {np.mean(loss_sup_set):.4f} | '
                    f'Smooth: {np.mean(loss_smooth_set):.4f} | '
                    f'scale: {smooth_scale:.2f}'
                )
                wandb.log({
                    'loss':             np.mean(loss_set),
                    'loss_supervised':  np.mean(loss_sup_set),
                    'loss_seg':         np.mean(loss_seg_set),
                    'loss_ch':          np.mean(loss_ch_set),
                    'loss_smooth':      np.mean(loss_smooth_set),
                    'smooth_scale':     smooth_scale,
                    'time':             timeit.default_timer() - start,
                    'step':             global_step,
                    'epoch':            epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_sup_set = [], [], []
                loss_smooth_set, loss_set = [], []

        assert epoch == epoch_float
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()

        model.save_model_latest(net, epoch, cfg)
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val >= best_f1_val:
            best_f1_val = f1_val
            wandb.log({'best val f1': best_f1_val, 'step': global_step, 'epoch': epoch_float})
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            model.save_model(net, epoch, cfg)

    net = model.load_model(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg  = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Running Point + Temporal Smoothness on device:', device)

    wandb.init(
        name=cfg.NAME + '_PointMono',
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
