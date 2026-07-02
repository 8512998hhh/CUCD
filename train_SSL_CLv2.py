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
# 序列内对比学习预训练
#   - 同一序列（同一地区）的不同时间步互为正样本
#   - 不同序列（不同地区）的时间步互为负样本
#   - 额外加入同帧双增强：同一时间、同一空间坐标互为正样本
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

    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    global_step = epoch_float = 0

    # 纯无监督，追踪最低的对比损失
    best_ssl_loss = float('inf')

    # 对比损失温度系数
    temperature = getattr(cfg.TRAINER, 'TEMPERATURE', 0.15)
    num_samples = getattr(cfg.TRAINER, 'NUM_SAMPLES', 512)
    lambda_temporal = getattr(cfg.TRAINER, 'LAMBDA_TEMPORAL', 1.0)
    lambda_aug = getattr(cfg.TRAINER, 'LAMBDA_AUG', 1.0)
    aug_temperature = getattr(cfg.TRAINER, 'AUG_TEMPERATURE', temperature)
    aug_brightness = getattr(cfg.TRAINER, 'AUG_BRIGHTNESS', 0.2)
    aug_contrast = getattr(cfg.TRAINER, 'AUG_CONTRAST', 0.2)

    print(
        f'CLv2: num_samples={num_samples}, temperature={temperature}, '
        f'lambda_temporal={lambda_temporal}, lambda_aug={lambda_aug}, '
        f'aug_temperature={aug_temperature}'
    )

    for epoch in range(1, epochs + 1):
        print(f'Starting Pre-training Epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        net.train()
        start = timeit.default_timer()
        loss_set, loss_temporal_set, loss_aug_set = [], [], []
        epoch_loss_set = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # 仅使用图像 x 和 timestamps，完全不使用标签
            x = batch['x'].to(device)
            timestamps = batch['timestamps'].to(device)
            point_idx = model.sample_balanced_point_indices(x, num_samples=num_samples)

            # =================================================================
            # 1) 时序相邻帧 InfoNCE：同坐标相邻帧为正样本，其它 batch 图像空间点为负样本
            # =================================================================
            loss_temporal = None
            if lambda_temporal > 0:
                features = net(x, edges, timestamps=timestamps, return_features=True, project=True)
                loss_temporal = model.info_nce_loss_dense_adjacent(
                    features,
                    temperature=temperature,
                    num_samples=num_samples,
                )
                (loss_temporal * lambda_temporal).backward()
                del features

            # =================================================================
            # 2) 同帧双增强 InfoNCE：同一帧同坐标双增强为正样本
            # =================================================================
            loss_aug = None
            if lambda_aug > 0:
                x_aug1 = model.color_augment_batch(
                    x,
                    brightness=aug_brightness,
                    contrast=aug_contrast,
                )
                x_aug2 = model.color_augment_batch(
                    x,
                    brightness=aug_brightness,
                    contrast=aug_contrast,
                )

                f_aug1 = net(x_aug1, edges, timestamps=timestamps, return_features=True, project=True)
                with torch.no_grad():
                    f_aug2 = net(x_aug2, edges, timestamps=timestamps, return_features=True, project=True)

                loss_aug = model.dense_aug_consistency_loss(
                    f_aug1,
                    f_aug2,
                    point_idx,
                    temperature=aug_temperature,
                    stop_grad_target=True,
                )
                (loss_aug * lambda_aug).backward()
                del f_aug1, f_aug2, x_aug1, x_aug2

            optimizer.step()

            loss_temporal_val = loss_temporal.item() if loss_temporal is not None else 0.0
            loss_aug_val = loss_aug.item() if loss_aug is not None else 0.0
            loss_val = lambda_temporal * loss_temporal_val + lambda_aug * loss_aug_val
            loss_set.append(loss_val)
            epoch_loss_set.append(loss_val)
            loss_temporal_set.append(loss_temporal_val)
            loss_aug_set.append(loss_aug_val)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                mean_loss = np.mean(loss_set)
                print(
                    f'Logging step {global_step} (epoch {epoch_float:.2f}) | '
                    f'SSL Loss: {mean_loss:.4f} | '
                    f'L_temporal: {np.mean(loss_temporal_set):.4f} | '
                    f'L_aug: {np.mean(loss_aug_set):.4f}'
                )

                time = timeit.default_timer() - start
                wandb.log({
                    'loss_ssl_contrastive': mean_loss,
                    'loss_temporal': np.mean(loss_temporal_set),
                    'loss_aug': np.mean(loss_aug_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, loss_temporal_set, loss_aug_set = [], [], []

        # ================== Epoch 结束 ==================
        if scheduler is not None:
            scheduler.step()

        current_epoch_loss = np.mean(epoch_loss_set) if len(epoch_loss_set) > 0 else 0

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
    cfg = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Runnning SSL Pre-training on device:', device)

    wandb.init(
        name=cfg.NAME + "_SSL_Pretrain",
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
