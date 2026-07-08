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
#   - 不做任何视图增强，单次前向传播，InfoNCE / SupCon 损失
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

    for epoch in range(1, epochs + 1):
        print(f'Starting Pre-training Epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        net.train()
        start = timeit.default_timer()
        loss_c_set = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # 仅使用图像 x 和 timestamps，完全不使用标签
            x = batch['x'].to(device)
            timestamps = batch['timestamps'].to(device)

            # =================================================================
            # 单次前向传播，提取时序特征 (B, T, C, H, W)
            # =================================================================
            features = net(x, edges, timestamps=timestamps, return_features=True, project=True)

            # =================================================================
            # 序列内对比损失：同序列时间步为正，跨序列为负
            # =================================================================
            loss = model.info_nce_loss_dense_adjacent(features, temperature=temperature, num_samples=num_samples)

            loss.backward()
            optimizer.step()

            loss_c_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                mean_loss = np.mean(loss_c_set)
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}) | SSL Loss: {mean_loss:.4f}')

                time = timeit.default_timer() - start
                wandb.log({
                    'loss_ssl_contrastive': mean_loss,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_c_set = []

        # ================== Epoch 结束 ==================
        if scheduler is not None:
            scheduler.step()

        current_epoch_loss = np.mean(loss_c_set) if len(loss_c_set) > 0 else 0

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
