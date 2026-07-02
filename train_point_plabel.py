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


def _expand_supervision_mask(loss_mask: torch.Tensor, num_slices: int) -> torch.Tensor:
    if loss_mask.dim() == 4:
        return loss_mask.unsqueeze(1).expand(-1, num_slices, -1, -1, -1)
    if loss_mask.dim() == 5 and loss_mask.shape[1] == 1 and num_slices != 1:
        return loss_mask.expand(-1, num_slices, -1, -1, -1)
    return loss_mask


@torch.no_grad()
def _build_confidence_pseudo_supervision(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    conf_thresh: float=0.95,
):
    """
    极简且绝对安全的伪标签生成函数。
    带有 @torch.no_grad() 和 .detach()，彻底隔绝梯度，防止 OOM 和图崩溃。
    """
    # 1. 彻底切断梯度
    logits = logits.detach()
    
    # 2. 形状与类型对齐
    loss_mask = _expand_supervision_mask(loss_mask.float(), logits.shape[1])
    target = target.float()

    # 3. 计算概率
    prob = torch.sigmoid(logits)

    # 4. 初始化伪标签掩膜与目标
    pseudo_mask = torch.zeros_like(loss_mask)
    pseudo_target = torch.zeros_like(target)

    # 5. 生成硬性规则：高于阈值为正样本，低于 (1-阈值) 为负样本，且绝不能覆盖真实掩膜
    is_pseudo_pos = (prob > conf_thresh) & (loss_mask == 0)
    is_pseudo_neg = (prob < (1.0 - conf_thresh)) & (loss_mask == 0)

    # 6. 赋值
    pseudo_mask[is_pseudo_pos] = 1.0
    pseudo_mask[is_pseudo_neg] = 1.0
    
    pseudo_target[is_pseudo_pos] = 1.0
    pseudo_target[is_pseudo_neg] = 0.0
    
    return pseudo_mask, pseudo_target


def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion_point = model.point_power_jaccard_loss
    criterion_edge = model.binary_edge_loss
    conf_thresh = getattr(cfg.TRAINER, 'PSEUDO_CONF_THRESH', 0.95)

    # reset the generators
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
        notice_flag = True
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})
        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            
            # 获取点监督标签和掩码
            y_point = batch['y_point'].to(device).float()
            loss_mask = batch['loss_mask'].to(device).float()
            y_edge = batch['y_edge'].to(device).float()
            
            # 仅执行一次前向传播
            logits_ch, logits_seg, logits_edge = net(x, edges)

            # 设定预热期 (强烈建议前 10 个 epoch 纯净训练)
            WARMUP_EPOCHS = 30

            # ==========================================
            # 1. 永远雷打不动的基石：真实点监督 Loss
            # ==========================================
            # 真实分割 Loss
            loss_seg_real = criterion_point(logits_seg, y_point, loss_mask)
            
            # 真实变化 Loss
            if 'y_ch' in batch:
                y_ch_point = batch['y_ch'].to(device).float()
                if y_ch_point.dim() == 4:
                    y_ch_point = y_ch_point.unsqueeze(2)
            else:
                y_ch_point = helpers.get_ch(y_point, edges).float()
            loss_ch_real = criterion_point(logits_ch, y_ch_point, loss_mask)

            # 真实边缘 Loss
            loss_edge = criterion_edge(logits_edge, y_edge)

            # ==========================================
            # 2. 伪标签扩展 Loss (安全区，独立计算，不会 OOM)
            # ==========================================
            loss_pseudo_seg = 0.0
            loss_pseudo_ch = 0.0

            if epoch > WARMUP_EPOCHS:
                # 获取分割的伪标签
                pseudo_mask_seg, pseudo_target_seg = _build_confidence_pseudo_supervision(
                    logits_seg, y_point, loss_mask, conf_thresh
                )
                
                # 获取变化的伪标签
                pseudo_mask_ch, pseudo_target_ch = _build_confidence_pseudo_supervision(
                    logits_ch, y_ch_point, loss_mask, conf_thresh
                )

                # --- 仅对伪标签掩码区域计算 BCE Loss (极致省显存) ---
                valid_idx_seg = (pseudo_mask_seg > 0.5)
                if valid_idx_seg.sum() > 0:
                    loss_pseudo_seg = F.binary_cross_entropy_with_logits(
                        logits_seg[valid_idx_seg], 
                        pseudo_target_seg[valid_idx_seg], 
                        reduction='mean'
                    )

                valid_idx_ch = (pseudo_mask_ch > 0.5)
                if valid_idx_ch.sum() > 0:
                    loss_pseudo_ch = F.binary_cross_entropy_with_logits(
                        logits_ch[valid_idx_ch], 
                        pseudo_target_ch[valid_idx_ch], 
                        reduction='mean'
                    )

                if notice_flag:
                    print(f"Pseudo labels: seg_pixels={valid_idx_seg.sum().item()}, ch_pixels={valid_idx_ch.sum().item()}")
                    notice_flag = False

            # ==========================================
            # 3. 最终汇总：让真实标签占据主导地位！
            # ==========================================
            # 给伪标签设置一个较小的权重，防止它带偏模型 (比如 0.1 到 0.5)
            PSEUDO_WEIGHT = 0.5 if epoch > WARMUP_EPOCHS else 0.0
            
            loss_seg = loss_seg_real + PSEUDO_WEIGHT * loss_pseudo_seg
            loss_ch = loss_ch_real + PSEUDO_WEIGHT * loss_pseudo_ch
            
            # 总损失
            loss = loss_seg + loss_ch + 0.5 * loss_edge
            
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

    print('=== Runnning on device:', device)

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