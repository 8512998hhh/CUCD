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
# 点监督 + dense 一致性 联合训练
#
# 主任务：
#   - 点监督建筑分割 loss_seg
#   - 点监督变化检测 loss_ch
#
# 辅助任务（联合训练，作用在投影头子空间 project=True，与主任务特征隔离）：
#   1) L_aug：同坐标双增强 dense 一致性。无需任何标签，是稠密预测里
#      最稳健的自监督信号，负样本只取“其它 AOI 的空间点”，不抓同序列其它帧。
#   2) L_tc：标签驱动的时序对比损失（InfoNCE）。只在“已标注的点”上，
#      用可靠的点标签把相邻帧分成稳定点 / 变化点：
#        - anchor / positive：稳定点在 t1 / t2 的同坐标特征（时序不变性）
#        - 负样本：其它稳定点 + 变化点特征（硬负样本，提供时序可分性）
#
# 设计要点（相对旧版的关键修正）：
#   - 旧版用变化头 sigmoid(logits_ch) 选 pull/push 像素。点监督下变化头
#     只在极稀疏点上训练过，整图变化概率不可信，等于持续注入噪声梯度。
#     这里改为只信任已标注点的真值标签，彻底去掉对变化头的依赖。
#   - 时序约束由 pull/push 两项合并成单个 InfoNCE：pull 不再无负样本坍缩，
#     变化点作硬负样本而非强推正交，强度由温度自动调节。
#   - 辅助损失全部走 proj_head（project=True），梯度只回流到
#     encoder/Mamba/decoder_seg 共享主干，不污染 outc_seg 输出层。
#   - 辅助权重保持较小，让点监督主任务主导优化。
# =====================================================================
def label_driven_temporal_contrastive_loss(
    features: torch.Tensor,
    y_point: torch.Tensor,
    loss_mask: torch.Tensor,
    edges,
    temperature: float = 0.1,
    num_samples: int = 4096,
):
    """
    标签驱动的时序对比损失（InfoNCE，只在已标注点上计算）。

    features:  (B, T, C, H, W)，来自 project=True 的投影空间
    y_point:   (B, T, 1, H, W)，每个标注点在各帧的真值（0/1）
    loss_mask: (B, 1, H, W)，1 = 该坐标已标注（跨帧共享）
    edges:     时间边；只用相邻边 (t1, t1+1)

    用一个 InfoNCE 取代旧版的 pull/push 两项：
      - anchor   = 时序稳定点（标签跨帧相同）在 t1 的特征
      - positive = 同一稳定点在 t2 的同坐标特征（时序不变性）
      - negative = ① 其它稳定点在 t2 的特征（不同地点，空间判别 + 抗坍缩）
                   ② 变化点（标签跨帧不同）的特征（硬负样本，等价于旧 push）

    相对 pull/push 分项的优点：
      - pull 不再是裸 1-cos（无负样本会坍缩），InfoNCE 自带负样本归一化；
      - 不再强推变化点正交，softmax 只要求正样本比负样本更近，由温度调强度；
      - 变化点没有可靠的“该靠近谁”信息，只作负样本、不给正目标，语义干净。

    变化点为空时退化为稳定点之间的 instance discrimination，仍然有效；
    稳定锚点为空时返回 0（无法构造正样本对）。
    """
    B, T, C, H, W = features.shape
    zero = features.sum() * 0.0

    annotated = loss_mask[:, 0] > 0.5            # (B, H, W)，标注坐标（跨帧共享）
    yp = y_point[:, :, 0] > 0.5                  # (B, T, H, W)，各帧真值

    stable_t1, stable_t2, change_feat = [], [], []
    for (t1, t2) in edges:
        if t2 != t1 + 1:
            continue
        if t1 < 0 or t2 >= T:
            continue

        same = (yp[:, t1] == yp[:, t2]) & annotated     # (B, H, W) 稳定点
        diff = (yp[:, t1] != yp[:, t2]) & annotated     # (B, H, W) 变化点

        if same.any():
            stable_t1.append(features[:, t1].permute(0, 2, 3, 1)[same])
            stable_t2.append(features[:, t2].permute(0, 2, 3, 1)[same])
        if diff.any():
            # 变化点的两帧特征都该被区分，一并放入硬负样本池
            change_feat.append(features[:, t1].permute(0, 2, 3, 1)[diff])
            change_feat.append(features[:, t2].permute(0, 2, 3, 1)[diff])

    if not stable_t1:
        return zero

    anchor = torch.cat(stable_t1, dim=0)
    positive = torch.cat(stable_t2, dim=0)
    n = anchor.size(0)
    if n > num_samples:
        idx = torch.randperm(n, device=features.device)[:num_samples]
        anchor = anchor[idx]
        positive = positive[idx]
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    Ns = anchor.size(0)

    # 正样本在对角线；其余稳定点的 t2 作为 batch 内负样本
    logits_parts = [anchor @ positive.t()]               # (Ns, Ns)

    if change_feat:
        neg = torch.cat(change_feat, dim=0)
        if neg.size(0) > num_samples:
            idx = torch.randperm(neg.size(0), device=features.device)[:num_samples]
            neg = neg[idx]
        neg = F.normalize(neg, dim=-1)
        logits_parts.append(anchor @ neg.t())            # (Ns, Nneg) 硬负样本

    logits = torch.cat(logits_parts, dim=1) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = torch.arange(Ns, device=features.device)
    return F.cross_entropy(logits, labels)


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

    # 辅助损失配置：默认保持较小，优先让点监督主任务主导优化。
    aux_num_samples = getattr(cfg.TRAINER, 'AUX_NUM_SAMPLES', getattr(cfg.TRAINER, 'NUM_SAMPLES', 512))
    # L_aug：同坐标双增强 dense 一致性（无需标签，最稳健的自监督信号）
    lambda_aug = getattr(cfg.TRAINER, 'LAMBDA_AUG', 0.1)
    # L_tc：标签驱动的时序对比损失（InfoNCE，只在已标注点上）
    lambda_tc = getattr(cfg.TRAINER, 'LAMBDA_TC',
                        getattr(cfg.TRAINER, 'LAMBDA_AUX_TEMPORAL', 0.1))
    aux_temperature = getattr(cfg.TRAINER, 'AUX_TEMPERATURE', 0.1)
    # 辅助损失 warmup：前若干 epoch 线性升温，避免早期不稳定特征污染主干
    aux_warmup_epochs = getattr(cfg.TRAINER, 'AUX_WARMUP_EPOCHS', 5)

    print(
        f'Point + SSL Aux: lambda_aug={lambda_aug}, lambda_tc={lambda_tc}, '
        f'aux_num_samples={aux_num_samples}, aux_temperature={aux_temperature}, '
        f'aux_warmup_epochs={aux_warmup_epochs}'
    )

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({
            'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })
        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_sup_set = [], [], []
        loss_aug_set, loss_tc_set, loss_set = [], [], []

        # 当前 epoch 的辅助损失升温系数
        if aux_warmup_epochs > 0:
            aux_scale = min(1.0, epoch / float(aux_warmup_epochs))
        else:
            aux_scale = 1.0

        for i, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_point = batch['y_point'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            timestamps = batch['timestamps'].to(device)

            # =============================================================
            # 1) 主任务：点监督分割 + 点监督变化检测
            # =============================================================
            logits_ch, logits_seg = net(x, edges, timestamps)
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
            loss_sup.backward()

            # =============================================================
            # 2) 辅助任务（联合训练，全部走投影头 project=True）
            #    a) L_aug：同坐标双增强 dense 一致性（无需标签）
            #    b) L_tc：标签驱动的时序对比损失（InfoNCE，只在已标注点上）
            # =============================================================
            loss_aug = loss_tc = None
            use_aug = lambda_aug > 0
            use_temporal = lambda_tc > 0

            if (use_aug or use_temporal) and aux_scale > 0:
                if use_aug:
                    # 两个独立颜色增强视图，几何位置不变 -> 同坐标为正样本对
                    x_aug1 = model.color_augment_batch(x)
                    x_aug2 = model.color_augment_batch(x)
                    f_aug1 = net(x_aug1, edges, timestamps=timestamps,
                                 return_features=True, project=True)
                    # f_aug2 仅作 dense_aug_consistency_loss 的 detach target（默认
                    # stop_grad_target=True），梯度从不回流。用 no_grad 跑这次前向，
                    # 立刻省掉一整张计算图的激活——峰值显存的主要来源。
                    with torch.no_grad():
                        f_aug2 = net(x_aug2, edges, timestamps=timestamps,
                                     return_features=True, project=True)
                    point_idx = model.sample_balanced_point_indices(x, num_samples=aux_num_samples)
                    loss_aug = model.dense_aug_consistency_loss(f_aug1, f_aug2, point_idx)
                    features = f_aug1  # 复用第一视图特征做时序对比，省一次前向
                else:
                    features = net(x, edges, timestamps=timestamps,
                                   return_features=True, project=True)

                if use_temporal:
                    loss_tc = label_driven_temporal_contrastive_loss(
                        features,
                        y_point,
                        loss_mask,
                        edges,
                        temperature=aux_temperature,
                        num_samples=aux_num_samples,
                    )

                loss_aux = features.sum() * 0.0
                if loss_aug is not None:
                    loss_aux = loss_aux + loss_aug * lambda_aug
                if loss_tc is not None:
                    loss_aux = loss_aux + loss_tc * lambda_tc
                (loss_aux * aux_scale).backward()
                del features

            optimizer.step()

            loss_aug_val = loss_aug.item() if loss_aug is not None else 0.0
            loss_tc_val = loss_tc.item() if loss_tc is not None else 0.0
            loss_val = loss_sup.item() + aux_scale * (
                lambda_aug * loss_aug_val
                + lambda_tc * loss_tc_val
            )

            loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_sup_set.append(loss_sup.item())
            loss_aug_set.append(loss_aug_val)
            loss_tc_set.append(loss_tc_val)
            loss_set.append(loss_val)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(
                    f'Logging step {global_step} (epoch {epoch_float:.2f}) | '
                    f'Loss: {np.mean(loss_set):.4f} | '
                    f'Sup: {np.mean(loss_sup_set):.4f} | '
                    f'Aug: {np.mean(loss_aug_set):.4f} | '
                    f'TC: {np.mean(loss_tc_set):.4f} | '
                    f'scale: {aux_scale:.2f}'
                )

                wandb.log({
                    'loss': np.mean(loss_set),
                    'loss_supervised': np.mean(loss_sup_set),
                    'loss_seg': np.mean(loss_seg_set),
                    'loss_ch': np.mean(loss_ch_set),
                    'loss_aug': np.mean(loss_aug_set),
                    'loss_tc': np.mean(loss_tc_set),
                    'aux_scale': aux_scale,
                    'time': timeit.default_timer() - start,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_sup_set = [], [], []
                loss_aug_set, loss_tc_set, loss_set = [], [], []

        assert epoch == epoch_float
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()

        model.save_model_latest(net, epoch, cfg)
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

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Running Point-Supervised Training with SSL Auxiliary Loss on device:', device)

    wandb.init(
        name=cfg.NAME + "_Point_SSL_Aux",
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
