import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import datasets_points, evaluation, experiment_manager, parsers, helpers
from model import model
from utils.helpers import active_learning_sampling, generate_active_labels_from_points
from utils.experiment_manager import CfgNode, save_experiment_config


def run_training(cfg: CfgNode):
    save_experiment_config(cfg)
    if cfg.TRAINER.PRETRAINED:
        net = model.load_model(cfg, device, pretrained=True)
        print("Pretrained Model Loaded.")
    else:
        net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion_point = model.point_power_jaccard_loss
    # reset the generators
    # 注意：这里必须使用 datasets_points 模块，而不是原来的 datasets 模块
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

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val = 0
    trigger_times = 0
    stop_training = False

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})
        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            
            # 获取点监督标签和掩码
            # 注意：在数据集中，包含变化标签时，y_ch 已经被生成了
            y_point = batch['y_point'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            timestamps = batch['timestamps'].to(device)
            logits_ch, logits_seg = net(x, edges, timestamps)

            # 分割损失使用点监督损失函数
            loss_seg = criterion_point(logits_seg, y_point, loss_mask)
            
            # 如果数据集中包含了变化标签 (include_change_label=True)
            if 'y_ch' in batch:
                # y_ch 的形状是 (B, E, H, W) 或 (B, E, 1, H, W)
                y_ch_dense = batch['y_ch'].to(device)
                
                # 确保 y_ch_dense 有通道维度 (B, E, 1, H, W)
                if y_ch_dense.dim() == 4:
                    y_ch_dense = y_ch_dense.unsqueeze(2)
                    
                # 使用点监督掩码过滤密集的变化标签，生成点监督的变化标签
                # loss_mask 的形状是 (B, 1, H, W)，会在 E 维度上广播
                y_ch_point = y_ch_dense * loss_mask.unsqueeze(1)
            else:
                # 如果数据集中没有生成 y_ch，则像之前一样通过 y_point 动态计算
                y_ch_point = helpers.get_ch(y_point, edges)

            # 使用点监督损失计算变化损失
            loss_ch = criterion_point(logits_ch, y_ch_point, loss_mask)
            
            loss = loss_seg + loss_ch
            loss.backward()
            optimizer.step()

            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
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
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_set = [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()
        # evaluation at the end of an epoch
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        # if f1_val <= best_f1_val:
        #     trigger_times += 1
        #     if trigger_times > cfg.TRAINER.PATIENCE:
        #         stop_training = True
        # else:
        #     best_f1_val = f1_val
        #     wandb.log({
        #         'best val f1': best_f1_val,
        #         'step': global_step,
        #         'epoch': epoch_float,
        #     })
        #     print(f'saving network (F1 {f1_val:.3f})', flush=True)
        #     model.save_model(net, epoch, cfg)
        #     trigger_times = 0

        # if stop_training:
        #     break

        if f1_val >= best_f1_val:
            best_f1_val = f1_val
            wandb.log({
                'best val f1': best_f1_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            model.save_model(net, epoch, cfg)

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
