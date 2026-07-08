"""
点监督 + 单调辅助头联合训练脚本
=================================================
主任务：点监督分割 (loss_seg) + 点监督变化检测 (loss_ch)
辅助任务：
  1) loss_mono   : 单调头的点监督（只对真值单次转变的标注点）
  2) loss_consist: 门控一致性（把自由头拉向单调头的干净曲线，多次变化像素自动退场）

配套模型: ContUrbanCDMambaModelMono / ContUrbanCDMambaModelCTMono
配置开关: MODEL.USE_MONOTONIC_AUX=True
"""
import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import datasets_points, evaluation, experiment_manager, parsers, helpers
from utils.experiment_manager import CfgNode, save_experiment_config
from model import model as base_model
from model.model_mono import (
    init_model_mono,
    load_model_mono,
    mono_point_loss,
    consistency_loss_gated,
)


def run_training(cfg: CfgNode):
    save_experiment_config(cfg)

    if getattr(cfg.TRAINER, 'PRETRAINED', False):
        net = load_model_mono(cfg, device, pretrained=True)
        print("Pretrained Model Loaded.")
    else:
        net = init_model_mono(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        if cfg.TRAINER.EPOCHS <= 1:
            return 1.0
        return 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion_point = base_model.point_power_jaccard_loss

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

    # ── 辅助损失超参（全部可通过 config 覆盖，默认值保守）─────────────────
    lambda_mono    = getattr(cfg.TRAINER, 'LAMBDA_MONO',    0.1)
    lambda_consist = getattr(cfg.TRAINER, 'LAMBDA_CONSIST', 0.1)
    mono_alpha     = getattr(cfg.TRAINER, 'MONO_ALPHA',     5.0)
    # warmup: 前 N epoch 辅助损失线性升温，避免早期随机特征污染主干
    warmup_epochs  = getattr(cfg.TRAINER, 'MONO_WARMUP_EPOCHS', 5)

    print(
        f"Mono Aux: lambda_mono={lambda_mono}, lambda_consist={lambda_consist}, "
        f"alpha={mono_alpha}, warmup={warmup_epochs}"
    )

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        start = timeit.default_timer()

        # 辅助损失升温系数
        aux_scale = min(1.0, epoch / float(warmup_epochs)) if warmup_epochs > 0 else 1.0

        loss_seg_set, loss_ch_set, loss_mono_set, loss_cons_set, loss_set = [], [], [], [], []

        for i, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            x          = batch['x'].to(device)
            y_point    = batch['y_point'].to(device)
            loss_mask  = batch['loss_mask'].to(device)
            timestamps = batch['timestamps'].to(device)

            # ── 主任务 + 单调头联合前向 ──────────────────────────────────
            logits_ch, logits_seg, p_mono = net(x, edges, timestamps, return_mono=True)

            # 1) 自由头点监督
            loss_seg = criterion_point(logits_seg, y_point, loss_mask)

            if 'y_ch' in batch:
                y_ch_dense = batch['y_ch'].to(device)
                if y_ch_dense.dim() == 4:
                    y_ch_dense = y_ch_dense.unsqueeze(2)
                y_ch_point = y_ch_dense * loss_mask.unsqueeze(1)
            else:
                y_ch_point = helpers.get_ch(y_point, edges)
            loss_ch = criterion_point(logits_ch, y_ch_point, loss_mask)

            loss_sup = loss_seg + loss_ch

            # 2) 单调头点监督（只在单次转变标注点上，门控在函数内部）
            loss_mono = mono_point_loss(p_mono, y_point, loss_mask)

            # 3) 门控一致性（warmup 后激活；梯度只流向自由头）
            p_free = torch.sigmoid(logits_seg)
            loss_consist = consistency_loss_gated(p_free, p_mono.detach(), alpha=mono_alpha)

            # ── 总损失 ───────────────────────────────────────────────────
            loss = (loss_sup
                    + lambda_mono * loss_mono
                    + aux_scale * lambda_consist * loss_consist)
            loss.backward()
            optimizer.step()

            # 记录
            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_mono_set.append(loss_mono.item())
            loss_cons_set.append(loss_consist.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                elapsed = timeit.default_timer() - start
                print(
                    f'Step {global_step} (ep {epoch_float:.2f}) | '
                    f'sup={np.mean(loss_seg_set)+np.mean(loss_ch_set):.4f} '
                    f'mono={np.mean(loss_mono_set):.4f} '
                    f'consist={np.mean(loss_cons_set):.4f} '
                    f'scale={aux_scale:.2f}'
                )
                wandb.log({
                    'loss':         np.mean(loss_set),
                    'loss_seg':     np.mean(loss_seg_set),
                    'loss_ch':      np.mean(loss_ch_set),
                    'loss_mono':    np.mean(loss_mono_set),
                    'loss_consist': np.mean(loss_cons_set),
                    'aux_scale':    aux_scale,
                    'time':         elapsed,
                    'step':         global_step,
                    'epoch':        epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_mono_set, loss_cons_set, loss_set = [], [], [], [], []

        assert epoch == epoch_float
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()

        base_model.save_model_latest(net, epoch, cfg)
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val >= best_f1_val:
            best_f1_val = f1_val
            wandb.log({'best val f1': best_f1_val, 'step': global_step, 'epoch': epoch_float})
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            base_model.save_model(net, epoch, cfg)

    # 最终测试集评估
    net = load_model_mono(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Running Point + Mono Aux Training on device:', device)

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
