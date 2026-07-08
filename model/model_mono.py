"""
Monotonic Auxiliary Head (层次3 双头一致性正则)
=======================================================
物理动机
--------
城市地块的演化本质是"长期稳定 + 偶发的一次方向性转变（建/拆）"。
本模块用一个单次转变参数化来显式建模这个物理先验：

    p_t^mono = p_start + (p_end - p_start) · σ(s·(t_real - τ))

  - τ     : 转变时刻（可学习，约束在时间跨度内）
  - p_start, p_end : 转变前/后的状态电平（编码方向：建→0→1，拆→1→0）
  - s     : 转变速度（可学习斜率，对应施工持续时长）

设计原则
--------
1. 主分支（自由头 outc_seg）负责表达和推理，保持任意变化模式的表达能力。
2. 单调辅助头仅在训练时出现，推理时完全不参与前向路径。
3. 通过残差门控（gated consistency loss），单调先验只在"局部呈单次转变"
   的像素上生效；多次/双向复杂变化像素门控自动关闭，不损害泛化性。
4. 全部代码在本文件内，不修改 model.py / modules.py / modules_ct.py。
   通过 init_model_mono() 初始化，config 开关控制是否启用。

与 CT 创新的协同
---------------
τ 使用真实 timestamps（相对月份），与 ContUrbanCDMambaModelCT 的物理时间
离散化在语义上一致——CT 负责输入侧时序建模，单调头负责输出侧时序约束。
"""

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Tuple, Sequence

from utils.experiment_manager import CfgNode
from model.model import (
    ContUrbanCDMambaModel,
    ContUrbanCDMambaModelCT,          # CT 版本也可作为基类
    load_state_dict_compatible,
    load_pretrained_encoder_mamba,
    save_model,
    save_model_latest,
    save_model_pretrained,
)


# =====================================================================
# Part 1: MonotonicAuxHead
# =====================================================================

class MonotonicAuxHead(nn.Module):
    """
    单次转变参数化辅助头。

    输入: decoder_seg 输出特征在 T 维平均后的空间图 (B, C, H, W)。
    输出: p_mono (B, T, 1, H, W)，由三个参数展开的单次转变曲线。

    参数空间（每像素3个标量）：
      - tau_raw  → τ  ∈ [t_min, t_max]  转变时刻
      - s_logit  → p_start ∈ [0,1]      转变前状态电平
      - e_logit  → p_end   ∈ [0,1]      转变后状态电平
      - log_s（全图共享）→ s 转变斜率（速度）
    """

    def __init__(self, in_channels: int, hidden_channels: int = None):
        super().__init__()
        hidden = hidden_channels or in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),   # 输出: tau_raw / start_logit / end_logit
        )
        # 可学习的斜率参数，初始化为 s=1.0（exp(0)=1）
        # s 越大转变越陡（快速建成），s 越小越平缓（慢慢施工）
        self.log_s = nn.Parameter(torch.zeros(1))

    def forward(self, feat_pooled: Tensor, t_abs: Tensor) -> Tensor:
        """
        feat_pooled : (B, C, H, W)  decoder 特征在时间维上的均值
        t_abs       : (B, T)        各帧的绝对相对时间（单位：月，首帧为0）
                      由 timestamps（delta_t）的 cumsum 得到
        返回:
            p_mono  : (B, T, 1, H, W)  单次转变曲线展开的每帧概率
        """
        B, C, H, W = feat_pooled.shape
        T = t_abs.shape[1]

        raw = self.conv(feat_pooled)              # (B, 3, H, W)
        tau_raw   = raw[:, 0:1]                   # (B, 1, H, W)
        s_logit   = raw[:, 1:2]
        e_logit   = raw[:, 2:3]

        # τ 约束在当前序列的时间跨度内（+margin），防止 τ 训飞
        t_min = t_abs.min(dim=1).values.view(B, 1, 1, 1)   # (B,1,1,1)
        t_max = t_abs.max(dim=1).values.view(B, 1, 1, 1)
        margin = (t_max - t_min).clamp(min=1.0) * 0.1
        tau = (t_min - margin) + (t_max - t_min + 2 * margin) * torch.sigmoid(tau_raw)

        p_start = torch.sigmoid(s_logit)          # (B, 1, H, W) 转变前状态
        p_end   = torch.sigmoid(e_logit)          # (B, 1, H, W) 转变后状态
        s = torch.exp(self.log_s)                 # 标量

        # 把 τ 和 t_abs 广播到相同形状进行展开
        # t_abs: (B, T) → (B, T, 1, 1, 1)
        t = t_abs.view(B, T, 1, 1, 1)
        # tau: (B, 1, H, W) → (B, 1, 1, H, W) 再广播到 (B, T, 1, H, W)
        tau_expand = tau.unsqueeze(1)             # (B, 1, 1, H, W)

        gate = torch.sigmoid(s * (t - tau_expand))    # (B, T, 1, H, W)

        p_start_e = p_start.unsqueeze(1)          # (B, 1, 1, H, W)
        p_end_e   = p_end.unsqueeze(1)

        p_mono = p_start_e + (p_end_e - p_start_e) * gate   # (B, T, 1, H, W)
        return p_mono


# =====================================================================
# Part 2: 损失函数
# =====================================================================

def count_transitions(y_seq: Tensor) -> Tensor:
    """
    统计每个像素在时间序列中发生状态转变的次数。
    y_seq: (B, T, H, W)  bool 或 0/1 float
    返回: (B, H, W)  int，每像素的转变次数
    """
    y_bool = y_seq > 0.5
    return (y_bool[:, 1:] != y_bool[:, :-1]).sum(dim=1)   # (B, H, W)


def mono_point_loss(
    p_mono: Tensor,
    y_point: Tensor,
    loss_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    单调头的点监督损失（Power Jaccard），只作用在"真值呈单次转变"的标注点上。

    p_mono    : (B, T, 1, H, W)  单调头预测概率
    y_point   : (B, T, 1, H, W)  点标注真值
    loss_mask : (B, 1, H, W)     原始标注掩码（1=有标注，0=无标注）

    逻辑：
      - 统计每个标注像素的 T 帧真值中，状态转变次数
      - 转变次数 <= 1 → 符合单次转变假设 → 允许参与单调头监督
      - 转变次数 > 1  → 不符合（被拆过又盖/复杂变化）→ 排除
    """
    # 统计转变次数，生成单调掩码
    y_seq = y_point[:, :, 0]                              # (B, T, H, W)
    n_trans = count_transitions(y_seq)                    # (B, H, W)
    is_mono = (n_trans <= 1).float().unsqueeze(1)         # (B, 1, H, W)

    # 组合：有标注 AND 单次转变
    combined_mask = loss_mask * is_mono                   # (B, 1, H, W)

    # 扩展掩码到 (B, T, 1, H, W) 与 p_mono 对齐
    if combined_mask.dim() == 4 and p_mono.dim() == 5:
        mask = combined_mask.unsqueeze(1).expand_as(p_mono)
    else:
        mask = combined_mask.expand_as(p_mono)

    iflat = p_mono[mask > 0.5]
    tflat = y_point[mask > 0.5]

    if iflat.numel() == 0:
        return (p_mono * 0).sum()

    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps
    return 1 - intersection / denom


def consistency_loss_gated(
    p_free: Tensor,
    p_mono: Tensor,
    alpha: float = 5.0,
) -> Tensor:
    """
    门控一致性损失：把自由头拉向单调头的干净曲线，但只对"单次转变像素"有效。

    p_free : (B, T, 1, H, W)  自由头输出（需要梯度，被拉的一方）
    p_mono : (B, T, 1, H, W)  单调头输出（stop-gradient，作为目标）
    alpha  : 门控锐度。越大 → 只有残差极小的像素才被拉。建议从 3~5 开始。

    门控机制：
      g = exp(-alpha · mean_T|p_free - p_mono|)
      - 自由头 ≈ 单调头（单次转变像素）：residual 小 → g≈1 → 强拉
      - 自由头 ≠ 单调头（多次/复杂变化）：residual 大 → g≈0 → 不拉

    梯度只流向 p_free（stop-gradient 施加在 p_mono 上），
    单调头不被噪声自由头带偏，各自有独立的监督来源。
    """
    # 在时间维度上求平均残差（detach 两者，纯统计量用于计算门控权重）
    resid = (p_free.detach() - p_mono.detach()).abs().mean(dim=1, keepdim=True)  # (B,1,1,H,W)
    g = torch.exp(-alpha * resid)                                                # (B,1,1,H,W)

    # 一致性损失：梯度只流向 p_free
    loss = (g * (p_free - p_mono.detach()) ** 2).mean()
    return loss


def change_consistency_from_mono(
    p_mono: Tensor,
    edges: Sequence[Tuple[int, int]],
) -> Tensor:
    """
    由单调曲线解析推导变化概率（供 change 头的软目标使用，可选）。

    p_mono : (B, T, 1, H, W)
    edges  : [(t1,t2), ...]
    返回: (B, E, 1, H, W)  每条边的预期变化概率
    """
    ch_list = []
    for t1, t2 in edges:
        delta = (p_mono[:, t2] - p_mono[:, t1]).abs()   # (B, 1, H, W)
        ch_list.append(delta)
    return torch.stack(ch_list, dim=1)                   # (B, E, 1, H, W)


# =====================================================================
# Part 3: _MonotonicMixin
# =====================================================================

class _MonotonicMixin:
    """
    为任意 ContUrbanCDMambaModel 变体注入单调辅助头的 Mixin。

    使用方法：放在 MRO 中 ContUrbanCDMambaModel（或其子类）之前，
    在子类 __init__ 里手动调用 _init_mono(cfg)。

    机制：
      - 在 decoder_seg 上注册 forward hook，每次 forward 执行时自动捕获
        decoder_seg 的输出特征，存入 self._cached_decoder_feats。
      - forward() 增加 return_mono=False 参数，为 True 时额外计算并返回
        p_mono；否则行为与父类完全一致。
      - inference() 与父类相同，只走自由头，不触发单调头（推理零开销）。
    """

    def _init_mono(self, cfg: CfgNode):
        in_ch = self.topology[0]                       # decoder 输出通道数
        self.mono_head = MonotonicAuxHead(in_ch)
        self._cached_decoder_feats: Tensor = None

        # 注册 hook：每次 decoder_seg 执行时缓存其输出
        def _hook(module, inp, output):
            self._cached_decoder_feats = output        # (B*T, C, H, W)

        self.decoder_seg.register_forward_hook(_hook)

    # ------------------------------------------------------------------
    def _compute_p_mono(self, timestamps: Tensor) -> Tensor:
        """
        由缓存的 decoder_seg 输出特征和 timestamps 计算 p_mono。

        self._cached_decoder_feats : (B*T, C, H, W)
        timestamps                 : (B, T, 1) delta_t

        返回: p_mono (B, T, 1, H, W)
        """
        assert self._cached_decoder_feats is not None, \
            "_compute_p_mono 必须在 forward() 之后调用"

        # 取出缓存特征，还原时间维度
        B_T, C, H, W = self._cached_decoder_feats.shape
        T = timestamps.shape[1]
        B = B_T // T
        feats = einops.rearrange(
            self._cached_decoder_feats, '(b t) c h w -> b t c h w', b=B
        )                                              # (B, T, C, H, W)

        # 在时间维取均值，得到空间特征图
        feat_pooled = feats.mean(dim=1)               # (B, C, H, W)

        # 计算绝对时间（delta → cumsum → 相对月份，首帧=0）
        ts = timestamps                                # (B, T, 1)
        if ts.dim() == 2:
            ts = ts.unsqueeze(-1)
        t_abs = ts.cumsum(dim=1).squeeze(-1)           # (B, T)

        return self.mono_head(feat_pooled, t_abs)      # (B, T, 1, H, W)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        edges: Sequence[Tuple[int, int]],
        timestamps: Tensor = None,
        return_features: bool = False,
        project: bool = False,
        return_mono: bool = False,
    ):
        """
        在父类 forward 基础上增加 return_mono 分支。
        return_mono=False 时行为与父类完全一致（传参透明）。
        """
        # 调用父类 forward（hook 会在此期间自动缓存 decoder_seg 输出）
        parent_result = super().forward(
            x, edges, timestamps,
            return_features=return_features,
            project=project,
        )

        if not return_mono:
            return parent_result

        # return_features=True 时父类只返回特征，不返回 logits，
        # 此时 mono 头无意义（训练时 return_mono 和 return_features 不同时使用）
        if return_features:
            return parent_result

        # parent_result = (logits_ch, logits_seg)
        logits_ch, logits_seg = parent_result
        p_mono = self._compute_p_mono(timestamps)
        return logits_ch, logits_seg, p_mono


# =====================================================================
# Part 4: 具体模型类（标准 Mamba + 单调头 / CT Mamba + 单调头）
# =====================================================================

class ContUrbanCDMambaModelMono(_MonotonicMixin, ContUrbanCDMambaModel):
    """
    标准 GLMamba + 单调辅助头。
    cfg.MODEL.USE_MAMBA=True, USE_CONTINUOUS_TIME=False, USE_MONOTONIC_AUX=True
    """
    def __init__(self, cfg: CfgNode):
        ContUrbanCDMambaModel.__init__(self, cfg)
        self._init_mono(cfg)
        print("Using GLMamba + MonotonicAuxHead")


class ContUrbanCDMambaModelCTMono(_MonotonicMixin, ContUrbanCDMambaModelCT):
    """
    连续时间 CT-GLMamba + 单调辅助头（两个创新点叠加）。
    cfg.MODEL.USE_MAMBA=True, USE_CONTINUOUS_TIME=True, USE_MONOTONIC_AUX=True
    """
    def __init__(self, cfg: CfgNode):
        ContUrbanCDMambaModelCT.__init__(self, cfg)
        self._init_mono(cfg)
        print("Using CT-GLMamba + MonotonicAuxHead")


# =====================================================================
# Part 5: 工厂函数与存取工具
# =====================================================================

def init_model_mono(cfg: CfgNode) -> nn.Module:
    """
    带单调辅助头的模型工厂。

    config 开关：
      MODEL.USE_MONOTONIC_AUX : True   启用单调辅助头（本文件）
      MODEL.USE_CONTINUOUS_TIME: True  同时叠加 CT 创新（默认 False）
      MODEL.USE_MAMBA          : True  必须为 True

    不改 model.py 中原有的 init_model()，本函数独立调用。
    """
    assert getattr(cfg.MODEL, 'USE_MAMBA', False), \
        "init_model_mono 要求 cfg.MODEL.USE_MAMBA=True"

    use_ct = getattr(cfg.MODEL, 'USE_CONTINUOUS_TIME', False)

    if use_ct:
        net = ContUrbanCDMambaModelCTMono(cfg)
    else:
        net = ContUrbanCDMambaModelMono(cfg)

    return torch.nn.DataParallel(net)


def load_model_mono(
    cfg: CfgNode,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    """
    初始化并加载权重（接口与 model.load_model 一致）。
    pretrained=True → 加载 pretrained.pt（encoder+Mamba 选择性迁移）
    pretrained=False → 加载 <NAME>.pt（完整权重）
    """
    net = init_model_mono(cfg)
    net.to(device)

    if pretrained:
        net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / 'pretrained.pt'
        checkpoint = torch.load(net_file, map_location=device)
        load_pretrained_encoder_mamba(net, checkpoint['weights'])
    else:
        net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
        checkpoint = torch.load(net_file, map_location=device)
        load_state_dict_compatible(net, checkpoint['weights'])

    print("load model from:", net_file)
    return net


def smart_load_model(
    cfg: CfgNode,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    """
    自动派发的模型加载函数，供 eval.py / inference.py 统一调用。

    根据 config 自动选择正确的加载路径：
      USE_MONOTONIC_AUX=True  → load_model_mono()  （本文件）
      否则                    → model.load_model()  （原 model.py）

    这样 eval.py / inference.py 只需把
        model.load_model(cfg, device)
    替换为
        smart_load_model(cfg, device)
    即可兼容所有模型变体，无需判断 config。
    """
    use_mono = getattr(cfg.MODEL, 'USE_MONOTONIC_AUX', False)
    if use_mono:
        return load_model_mono(cfg, device, pretrained=pretrained)
    else:
        from model.model import load_model as _load_model
        return _load_model(cfg, device, pretrained=pretrained)
