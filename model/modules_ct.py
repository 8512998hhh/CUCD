"""
Continuous-Time (CT) Selective State Space modules.

创新点：连续时间 SSM 离散化。
--------------------------------------------------------------------------
标准 Mamba/S6 的离散步长 Δ 完全由数据驱动（dt_proj(x)），隐含假设时间步
均匀。而卫星影像的重访间隔本身是不规则的物理时间。本模块把 SSM 的离散步长
与真实物理时间间隔 Δt 绑定：

        Δ_eff = softplus(dt_proj(x) + b) · Δt_norm

其中 Δt_norm 是归一化后的相邻帧物理间隔（来自 dataset 的 delta_t）。
这样"间隔 6 个月"与"间隔 1 个月"在状态演化 A_bar = exp(Δ_eff · A) 与
输入增益 B_bar 上自然拉开差距，是 SSM 处理不规则采样时间序列的理论正确形式。

本文件所有类均为新增，不修改 modules.py 中任何现有实现。原 SS1D_TIG 的
乘性时间门控（x = x + x*sigmoid(MLP(ts))）在这里被移除，改由离散化步长承载
时间信息，两者互不干扰。
"""

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 复用 modules.py 中已实现的底层算子，保持与主干完全一致的数值行为
from model.modules import (
    SelectiveScanCore,
    selective_scan_ref,
    CrossScan1D,
    CrossMerge1D,
    selective_scan_cuda_core,
)


def cross_selective_scan_1d_ct(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    x_proj_bias: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    A_logs: torch.Tensor,
    Ds: torch.Tensor,
    delta_t: torch.Tensor = None,          # 🚀 新增：物理时间间隔 (B, L)，已归一化
    out_norm: torch.nn.Module = None,
    nrows: int = -1,
    backnrows: int = -1,
    ssoflex: bool = True,
    SelectiveScan=SelectiveScanCore,
    CrossScan=CrossScan1D,
    CrossMerge=CrossMerge1D,
):
    """
    连续时间版本的双向 1D selective scan。

    与 modules.cross_selective_scan_1d 的唯一区别：
      - 步长 Δ 在送入 scan 之前，先在 torch 侧完成 softplus(dt + bias)，
        再逐时间步乘以物理间隔 delta_t；
      - 因此调用底层 scan 时 delta_softplus=False、delta_bias=0，避免二次 softplus。

    x:        (B, D, L)
    delta_t:  (B, L) 或 None。None 时退化为均匀间隔（全 1），等价标准 Mamba。
    """
    B, D, L = x.shape
    D_state = A_logs.shape[1]
    K, D_inner, R = dt_projs_weight.shape  # K=2 双向
    N = D_state

    # 1. 双向展开
    if CrossScan is not None:
        xs = CrossScan.apply(x)  # (B, K, D, L)
    else:
        xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)

    # 2. 投影出 dt / B / C
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)

    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)  # (B, K, D, L)

    # 3. 🚀 连续时间离散化：softplus(dt + bias) · Δt
    #    dt_projs_bias: (K, D) -> (1, K, D, 1)
    bias = dt_projs_bias.view(1, K, D_inner, 1)
    dts = F.softplus(dts + bias)  # 在 torch 侧完成 softplus

    if delta_t is not None:
        # delta_t: (B, L) -> (B, 1, 1, L)，在 K、D 维广播
        dt_phys = delta_t.view(B, 1, 1, L).to(dts.dtype)
        dts = dts * dt_phys

    # 4. 整理成底层 scan 期望的形状 (B, K*D, L)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)

    As = -torch.exp(A_logs.to(torch.float))           # (K*D, N)
    Ds = Ds.to(torch.float)                            # (K*D)
    # softplus 与时间缩放已在上面完成 -> 关闭内核 softplus，偏置置零
    zero_bias = torch.zeros(K * D_inner, device=x.device, dtype=torch.float)

    def run_scan(u, delta, A, Bm, Cm, Dm, db):
        if selective_scan_cuda_core is not None:
            return SelectiveScan.apply(u, delta, A, Bm, Cm, Dm, db, False, nrows, backnrows, ssoflex)
        # CPU / 无 CUDA 回退：逐方向调用参考实现
        ys_list = []
        xs_rs = u.view(B, K, D_inner, L)
        dts_rs = delta.view(B, K, D_inner, L)
        Bs_rs = Bm.view(B, K, N, L)
        Cs_rs = Cm.view(B, K, N, L)
        for k in range(K):
            y_k = selective_scan_ref(
                xs_rs[:, k], dts_rs[:, k],
                A[k * D_inner:(k + 1) * D_inner, :],
                Bs_rs[:, k], Cs_rs[:, k],
                Dm[k * D_inner:(k + 1) * D_inner],
                None, delta_softplus=False,
            )
            ys_list.append(y_k)
        return torch.stack(ys_list, dim=1).view(B, K * D_inner, L)

    ys = run_scan(xs, dts, As, Bs, Cs, Ds, zero_bias).view(B, K, -1, L)

    # 5. 双向合并
    if CrossMerge is not None:
        y = CrossMerge.apply(ys)
    else:
        y = ys[:, 0] + torch.flip(ys[:, 1], dims=[-1])

    if out_norm is not None:
        y = y.transpose(1, 2)
        y = out_norm(y)
        y = y.transpose(1, 2)

    return y


class SS1D_CT(nn.Module):
    """
    连续时间 selective scan 模块（替代 SS1D_TIG）。

    与 SS1D_TIG 的差异：
      - 去掉 time_gate_mlp 的乘性门控；
      - 时间信息通过 cross_selective_scan_1d_ct 的 delta_t 注入离散化步长。

    参数命名与 SS1D_TIG 保持一致（in_proj / conv1d / x_proj_weight /
    dt_projs_weight / dt_projs_bias / A_logs / Ds / out_norm / out_proj），
    因此如需从已训练的 SS1D_TIG 权重迁移，同名张量可直接加载。
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()

        if d_conv >= 1:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        k_group = 2
        self.x_proj_weight = nn.Parameter(torch.empty(k_group, self.dt_rank + d_state * 2, self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.empty(k_group, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(k_group, self.d_inner))
        self.A_logs = nn.Parameter(torch.empty(k_group * self.d_inner, d_state))
        self.Ds = nn.Parameter(torch.ones(k_group * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self._init_params(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

    def _init_params(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt.repeat(2).reshape(2, -1))

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        with torch.no_grad():
            self.A_logs.copy_(A_log.repeat(2, 1))

    def forward(self, x, ts=None):
        """
        x:  (B, L, D)
        ts: (B, L, 1) 或 (B, L) 归一化后的物理时间间隔；None 时退化为标准 Mamba。
        """
        B, L, _ = x.shape
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)

        if self.d_conv >= 1:
            x = x.transpose(1, 2)
            x = self.conv1d(x)
            x = x.transpose(1, 2)

        x = self.act(x)
        z = self.act(z)

        # 归一化时间间隔整理为 (B, L)
        delta_t = None
        if ts is not None:
            delta_t = ts.squeeze(-1) if ts.dim() == 3 else ts  # (B, L)

        x_ssm = x.transpose(1, 2).contiguous()  # (B, D, L)
        y = cross_selective_scan_1d_ct(
            x_ssm,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            delta_t=delta_t,
            out_norm=self.out_norm,
        )

        y = y.transpose(1, 2)  # (B, L, D)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class GLSSM_CT(nn.Module):
    """全局 + 局部窗口的连续时间 SSM（结构对应 GLSSM，内部用 SS1D_CT）。"""

    def __init__(
        self,
        d_model,
        d_state=16,
        ssm_ratio=2.0,
        act_layer=nn.SiLU,
        d_conv=1,
        window_size=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.g_ssm = SS1D_CT(d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio / 2,
                             d_conv=d_conv, act_layer=act_layer, **kwargs)
        self.l_ssm = SS1D_CT(d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio / 2,
                             d_conv=d_conv, act_layer=act_layer, **kwargs)
        self.window_size = window_size
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_act = nn.Sigmoid()
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, ts=None):
        B, L, D = x.shape

        # 全局分支
        out_g = self.g_ssm(x, ts)

        # 局部窗口分支
        K = self.window_size if self.window_size is not None else max(1, L // 2)
        pad_len = (K - L % K) % K
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
            ts_pad = F.pad(ts, (0, 0, 0, pad_len)) if ts is not None else None
        else:
            x_pad = x
            ts_pad = ts

        x_local = einops.rearrange(x_pad, 'b (n k) d -> (b n) k d', k=K)
        if ts_pad is not None:
            ts_local = einops.rearrange(ts_pad, 'b (n k) c -> (b n) k c', k=K)
        else:
            ts_local = None

        out_l_local = self.l_ssm(x_local, ts_local)
        out_l_pad = einops.rearrange(out_l_local, '(b n) k d -> b (n k) d', b=B)
        out_l = out_l_pad[:, :L, :] if pad_len > 0 else out_l_pad

        gate = self.gate_act(self.gate_linear(x))
        out = out_g * gate + out_l * gate
        return self.out_linear(out)


class GLMambaTFRModuleCT(nn.Module):
    """
    连续时间版本的时序特征精炼模块（结构对应 GLMambaTFRModule）。

    与原模块的接口完全一致：forward(features, timestamps)，
    features: (B, T, D, H, W)，timestamps: (B, T) 或 (B, T, 1)。

    区别：不再把 timestamps 当作乘性门控输入，而是先做归一化，再作为
    连续时间离散化步长注入 GLSSM_CT。
    """

    def __init__(self, t: int, d_model: int, n_layers: int = 1, d_state: int = 16,
                 window_size: int = 6, ts_scale: float = 1.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        # 时间间隔归一化尺度：delta_t(月) / ts_scale。默认 1.0 表示以"月"为单位。
        # 若物理间隔偏大导致 Δ_eff 过大、状态衰减过快，可调大 ts_scale。
        self.ts_scale = ts_scale

        self.layers = nn.ModuleList([
            GLSSM_CT(d_model=d_model, d_state=d_state, window_size=window_size, **kwargs)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, features: Tensor, timestamps: Tensor = None) -> Tensor:
        B, T, D, H, W = features.size()
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')

        if timestamps is not None:
            if timestamps.dim() == 2:
                timestamps = timestamps.unsqueeze(-1)      # (B, T, 1)
            # 归一化物理间隔
            ts_norm = timestamps.to(features.dtype) / self.ts_scale
            ts_expanded = ts_norm.unsqueeze(1).unsqueeze(2).expand(B, H, W, T, 1)
            ts_tokens = einops.rearrange(ts_expanded, 'B H W T C -> (B H W) T C')
        else:
            ts_tokens = None

        chunk_size = 4096
        tokens_out = []
        for i in range(0, tokens.size(0), chunk_size):
            chunk = tokens[i:i + chunk_size]
            ts_chunk = ts_tokens[i:i + chunk_size] if ts_tokens is not None else None
            for layer, norm in zip(self.layers, self.norms):
                out = layer(chunk, ts_chunk)
                chunk = norm(chunk + out)
            tokens_out.append(chunk)

        features_hat = torch.cat(tokens_out, dim=0)
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)
        return features_hat
