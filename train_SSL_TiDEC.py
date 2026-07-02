import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import experiment_manager, parsers, helpers
from utils import datasets_points as datasets
from model import model
from utils.experiment_manager import CfgNode


# =====================================================================
# TiDEC 无监督预训练
#   L_aug    : 同帧双增强 dense 一致性
#   L_stable : 时序稳定点上的 long-term 一致性（无建筑 mask）
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
    best_ssl_loss = float('inf')

    num_samples = getattr(cfg.TRAINER, 'NUM_SAMPLES', 512)
    lambda_aug = getattr(cfg.TRAINER, 'LAMBDA_AUG', 1.0)
    lambda_stable = getattr(cfg.TRAINER, 'LAMBDA_STABLE', 1.0)
    stable_ratio = getattr(cfg.TRAINER, 'STABLE_RATIO', 0.5)
    aug_brightness = getattr(cfg.TRAINER, 'AUG_BRIGHTNESS', 0.2)
    aug_contrast = getattr(cfg.TRAINER, 'AUG_CONTRAST', 0.2)
    aug_temperature = getattr(cfg.TRAINER, 'AUG_TEMPERATURE', 0.15)

    print(
        f'TiDEC CL (low-memory): num_samples={num_samples}, '
        f'lambda_aug={lambda_aug}, lambda_stable={lambda_stable}, '
        f'stable_ratio={stable_ratio}, aug_temperature={aug_temperature}'
    )
    print('Memory mode: sequential forward + stop-grad aug target (peak ~1x batch forward).')

    for epoch in range(1, epochs + 1):
        print(f'Starting TiDEC Pre-training Epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        net.train()
        start = timeit.default_timer()
        loss_set, loss_aug_set, loss_stable_set = [], [], []
        epoch_loss_set = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            timestamps = batch['timestamps'].to(device)
            point_idx = model.sample_balanced_point_indices(x, num_samples=num_samples)

            loss_aug = loss_stable = None

            # 1) L_stable：单独前向 + backward，完成后释放激活
            if lambda_stable > 0:
                features = net(x, edges, timestamps=timestamps, return_features=True, project=False)
                loss_stable = model.dense_stable_temporal_loss(
                    features, point_idx, stable_ratio=stable_ratio,
                )
                (loss_stable * lambda_stable).backward()
                del features

            # 2) L_aug：两次 B 前向（非 2B），目标分支 stop-grad，进一步省显存
            if lambda_aug > 0:
                x_aug1 = model.color_augment_batch(
                    x, brightness=aug_brightness, contrast=aug_contrast,
                )
                x_aug2 = model.color_augment_batch(
                    x, brightness=aug_brightness, contrast=aug_contrast,
                )
                f_aug1 = net(x_aug1, edges, timestamps=timestamps, return_features=True,project=True)
                with torch.no_grad():
                    f_aug2 = net(x_aug2, edges, timestamps=timestamps, return_features=True,project=True)
                loss_aug = model.dense_aug_consistency_loss(
                    f_aug1, f_aug2, point_idx,
                    temperature=aug_temperature,
                    stop_grad_target=True,
                )
                (loss_aug * lambda_aug).backward()
                del f_aug1, f_aug2, x_aug1, x_aug2

            optimizer.step()

            loss_aug_val = loss_aug.item() if loss_aug is not None else 0.0
            loss_stable_val = loss_stable.item() if loss_stable is not None else 0.0
            loss_val = lambda_aug * loss_aug_val + lambda_stable * loss_stable_val
            loss_set.append(loss_val)
            epoch_loss_set.append(loss_val)
            loss_aug_set.append(loss_aug_val)
            loss_stable_set.append(loss_stable_val)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(
                    f'Logging step {global_step} (epoch {epoch_float:.2f}) | '
                    f'Loss: {np.mean(loss_set):.4f} | '
                    f'L_aug: {np.mean(loss_aug_set):.4f} | '
                    f'L_stable: {np.mean(loss_stable_set):.4f}'
                )
                wandb.log({
                    'loss_ssl_tidex': np.mean(loss_set),
                    'loss_aug': np.mean(loss_aug_set),
                    'loss_stable': np.mean(loss_stable_set),
                    'time': timeit.default_timer() - start,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, loss_aug_set, loss_stable_set = [], [], []

        if scheduler is not None:
            scheduler.step()

        current_epoch_loss = np.mean(epoch_loss_set) if len(epoch_loss_set) > 0 else 0.0

        model.save_model_latest(net, epoch, cfg)
        print(f'Saved latest network (Epoch {epoch})', flush=True)

        if current_epoch_loss < best_ssl_loss:
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
    print('=== Running TiDEC SSL Pre-training on device:', device)

    wandb.init(
        name=cfg.NAME + "_SSL_TiDEC",
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
