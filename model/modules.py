import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops

import numpy as np

# https://pgmpy.org/models/markovnetwork.html
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

from typing import Tuple, Sequence, Callable, Optional, Any

from joblib import Parallel, delayed

import os
import math
import copy
from functools import partial

os.environ["PYTHONWARNINGS"] = "ignore"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# VMamba / Mamba Core Components (Adapted from VMamba project)
# =============================================================================

# Try importing CUDA kernels, but provide a fallback if they are missing
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None

try:
    import selective_scan_cuda_core
except ImportError:
    selective_scan_cuda_core = None

try:
    import selective_scan_cuda_oflex
except ImportError:
    selective_scan_cuda_oflex = None


# Pure PyTorch Reference Implementation for CPU/Debug
def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
        
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    if A.dim() == 3:
        A = A[0] # assuming sharing

    # x: (B, D, N)
    x = torch.zeros((batch, dim, dstate), device=u.device)
    
    ys = []
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
         deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
         deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
         
    # Sequential scan
    # x_t = A_bar * x_{t-1} + B_bar * u_t
    # y_t = C * x_t + D * u_t
    
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, :, i]
        if not is_variable_C:
             y = torch.einsum('bdn,dn->bd', x, C)
        else:
             y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
             
        if i == u.shape[2] - 1 and return_last_state:
            last_state = x
            
        ys.append(y)
        
    y = torch.stack(ys, dim=2) # (B, D, L)
    
    if D is not None:
        y = y + u * D[..., None]
        
    return y if not return_last_state else (y, last_state)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        
        # Fallback to pure PyTorch if CUDA kernel not available or fails
        if selective_scan_cuda_core is None:
             # Use reference implementation (Slow but works on CPU/No-CUDA)
             # Reshape B and C if necessary to match reference implementation expectations
             # Core usually expects B, C as (B, N, L) or (B, K, N, L) depending on group
             # print("Using reference implementation for selective scan")
             return selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus)
        
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        # Fallback for backward pass
        if selective_scan_cuda_core is None:
             raise NotImplementedError("Backward pass for SelectiveScanRef is not implemented in this snippet. Please use CUDA for training.")

        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# 1D Cross Scan for Time: Forward and Backward
class CrossScan1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x: (B, C, L)
        B, C, L = x.shape
        ctx.shape = (B, C, L)
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(x, dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # ys: (B, 2, C, L)
        B, C, L = ctx.shape
        # Sum gradients from forward and backward streams
        # forward gradient: ys[:, 0]
        # backward gradient: flip(ys[:, 1])
        y = ys[:, 0] + torch.flip(ys[:, 1], dims=[-1])
        return y


class CrossMerge1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # ys: (B, 2, C, L)
        B, K, C, L = ys.shape
        ctx.shape = (B, C, L)
        # Sum outputs: Forward + Flip(Backward)
        y = ys[:, 0] + torch.flip(ys[:, 1], dims=[-1])
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # x: (B, C, L)
        # Gradients are distributed to both streams
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(x, dims=[-1])
        return xs


def cross_selective_scan_1d(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    # ==============================
    nrows = -1, 
    backnrows = -1, 
    ssoflex=True,
    # ==============================
    SelectiveScan=SelectiveScanCore,
    CrossScan=CrossScan1D,
    CrossMerge=CrossMerge1D,
):
    # x: (B, D, L)
    B, D, L = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape # K=2 for bidirectional

    # 1. Project x to input parameters (delta, B, C)
    # x_dbl: (B, K, D_total, L) where D_total = R + N + N
    # Using efficient conv1d implementation approach from VMamba
    
    if CrossScan is not None:
         xs = CrossScan.apply(x) # (B, K, D, L)
    else:
         # Fallback if custom autograd not available
         xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
    
    # Project input x to delta_rank + 2*d_state
    # We do this for each direction (K=2)
    # x_proj_weight: (K*D, rank+2*state) ? No, VMamba uses (K, D_out, D_in) equivalent logic
    
    # Implementing "no_einsum" logic for speed usually, but sticking to einsum for clarity if needed
    # x_proj_weight in VMamba: (K, N_out, D)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    
    # Project delta
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    # Prepare for selective scan
    xs = xs.view(B, -1, L)           # (B, K*D, L)
    dts = dts.contiguous().view(B, -1, L) # (B, K*D, L)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    
    As = -torch.exp(A_logs.to(torch.float)) # (K*D, N)
    Ds = Ds.to(torch.float)                 # (K*D)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    # Run Selective Scan
    # We use the Core implementation
    # Note: We need to define selective_scan wrapper or use apply directly
    def run_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
        if selective_scan_cuda_core is not None:
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
        else:

            B_prime = B * K
            xs_f = xs.view(B_prime, D, L)
            dts_f = dts.view(B_prime, D, L)
            As_f = As.view(K, D, N).repeat_interleave(B, dim=0).view(B_prime, D, N) # Wait, A is parameter, not batched
          
            ys_list = []
            for k in range(K):
                
                u_k = x_dbl[:, k, :D, :] # Wait, xs is separate from x_dbl?

                
                # Recalculate xs_k
                if CrossScan is not None:
                     # Re-run CrossScan locally or assume we have access
                     # xs is (B, K*D, L). We can reshape.
                     xs_rs = xs.view(B, K, D, L)
                     u_k = xs_rs[:, k, :, :]
                     
                     dts_rs = dts.view(B, K, D, L)
                     delta_k = dts_rs[:, k, :, :]
                     
                     Bs_rs = Bs.view(B, K, N, L)
                     B_k = Bs_rs[:, k, :, :]
                     
                     Cs_rs = Cs.view(B, K, N, L)
                     C_k = Cs_rs[:, k, :, :]
                     
                     # A is (K*D, N). 
                     A_k = As[k*D : (k+1)*D, :]
                     D_k = Ds[k*D : (k+1)*D]
                     db_k = delta_bias[k*D : (k+1)*D]
                     
                     y_k = selective_scan_ref(u_k, delta_k, A_k, B_k, C_k, D_k, db_k, delta_softplus)
                     ys_list.append(y_k)
                     
            return torch.stack(ys_list, dim=1).view(B, K*D, L) # Match CUDA output shape (B, K, D, L)? No CUDA output is (B, K*D, L) from scan? 
            # Wait, selective_scan_cuda usually returns (B, D, L) or (B, K*D, L)
            # Our code above: ys = run_scan(...).view(B, K, -1, L)
            # So run_scan must return (B, K*D, L) flattened.
            
            return torch.stack(ys_list, dim=1).view(B, K*D, L)

    ys = run_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, L)
    
    # Merge results
    if CrossMerge is not None:
        y = CrossMerge.apply(ys) # (B, D, L)
    else:
        # Fallback
        y = ys[:, 0] + torch.flip(ys[:, 1], dims=[-1])
    
    if out_norm is not None:
        # out_norm expects (B, L, C) usually or (B, C, L) depending on impl
        # Here we assume (B, D, L) input, LayerNorm usually wants (..., D)
        y = y.transpose(1, 2) # (B, L, D)
        y = out_norm(y)
        y = y.transpose(1, 2) # (B, D, L)
        
    return y


class SS1D(nn.Module):
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
        
        # In Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        
        # Conv1D (Depthwise)
        if d_conv >= 1:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )
        
        # SSM Parameters
        k_group = 2 # Bidirectional (Forward, Backward)
        
        # x_proj: Project input to dt, B, C
        # Weight shape: (K, R+N+N, D_inner)
        self.x_proj_weight = nn.Parameter(torch.empty(k_group, self.dt_rank + d_state * 2, self.d_inner))
        
        # dt_proj: Project dt_rank to d_inner
        self.dt_projs_weight = nn.Parameter(torch.empty(k_group, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(k_group, self.d_inner))
        
        # A, D
        self.A_logs = nn.Parameter(torch.empty(k_group * self.d_inner, d_state))
        self.Ds = nn.Parameter(torch.ones(k_group * self.d_inner))
        
        # Out Projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self._init_params(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

    def _init_params(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        # Initialize weights (Simplified version of VMamba init)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        
        # Initialize dt bias
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
             self.dt_projs_bias.copy_(inv_dt.repeat(2).reshape(2, -1)) # K=2

        # A logs
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        with torch.no_grad():
             self.A_logs.copy_(A_log.repeat(2, 1)) # K=2

    def forward(self, x, ts=None):
        # x: (B, L, D)
        B, L, _ = x.shape
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (B, L, D_inner)
        
        # Conv1D
        if self.d_conv >= 1:
            x = x.transpose(1, 2) # (B, D, L)
            x = self.conv1d(x)
            x = x.transpose(1, 2) # (B, L, D)
            
        x = self.act(x)
        z = self.act(z)
        
        # SSM Core
        # Prepare for cross_selective_scan_1d which expects (B, D, L)
        x_ssm = x.transpose(1, 2).contiguous()
        
        y = cross_selective_scan_1d(
            x_ssm, 
            self.x_proj_weight, 
            None, 
            self.dt_projs_weight, 
            self.dt_projs_bias,
            self.A_logs, 
            self.Ds,
            delta_softplus=True,
            out_norm=self.out_norm
        )
        
        # y is (B, D, L)
        y = y.transpose(1, 2) # (B, L, D)
        
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class SS1D_TIG(nn.Module):
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
        
        # In Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        
        # Conv1D (Depthwise)
        if d_conv >= 1:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )
        
        # SSM Parameters
        k_group = 2 # Bidirectional (Forward, Backward)
        
        # x_proj: Project input to dt, B, C
        # Weight shape: (K, R+N+N, D_inner)
        self.x_proj_weight = nn.Parameter(torch.empty(k_group, self.dt_rank + d_state * 2, self.d_inner))
        
        # dt_proj: Project dt_rank to d_inner
        self.dt_projs_weight = nn.Parameter(torch.empty(k_group, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(k_group, self.d_inner))
        
        # A, D
        self.A_logs = nn.Parameter(torch.empty(k_group * self.d_inner, d_state))
        self.Ds = nn.Parameter(torch.ones(k_group * self.d_inner))
        
        # Out Projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self._init_params(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)
        
        self.time_gate_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, self.d_inner)
        )

    def _init_params(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        # Initialize weights (Simplified version of VMamba init)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        
        # Initialize dt bias
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
             self.dt_projs_bias.copy_(inv_dt.repeat(2).reshape(2, -1)) # K=2

        # A logs
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        with torch.no_grad():
             self.A_logs.copy_(A_log.repeat(2, 1)) # K=2

    def forward(self, x, ts=None):
        # x: (B, L, D)
        B, L, _ = x.shape
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (B, L, D_inner)
        
        # Conv1D
        if self.d_conv >= 1:
            x = x.transpose(1, 2) # (B, D, L)
            x = self.conv1d(x)
            x = x.transpose(1, 2) # (B, L, D)
            
        x = self.act(x)
        z = self.act(z)
        
        # SSM Core
        # Prepare for cross_selective_scan_1d which expects (B, D, L)
        if ts is not None:
            if ts.dim() == 2:
                ts = ts.unsqueeze(-1)
            # 计算 0~1 的时间门控权重
            time_gate = torch.sigmoid(self.time_gate_mlp(ts)) # (B, L, D_inner)
            x = x + x * time_gate
        x_ssm = x.transpose(1, 2).contiguous()
        
        y = cross_selective_scan_1d(
            x_ssm, 
            self.x_proj_weight, 
            None, 
            self.dt_projs_weight, 
            self.dt_projs_bias,
            self.A_logs, 
            self.Ds,
            delta_softplus=True,
            out_norm=self.out_norm
        )
        
        # y is (B, D, L)
        y = y.transpose(1, 2) # (B, L, D)
        
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class GLSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=1,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        window_size=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Branch 1: Global SSM
        self.g_ssm = SS1D_TIG(
            d_model=d_model, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio/2, # Half capacity for Global
            d_conv=d_conv,
            act_layer=act_layer,
            **kwargs
        )
        
        # Branch 2: Local SSM
        self.l_ssm = SS1D_TIG(
            d_model=d_model, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio/2, # Half capacity for Local
            d_conv=d_conv,
            act_layer=act_layer,
            **kwargs
        )
        
        # Window size for Local SSM
        self.window_size = window_size
        
        # Gating branch
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_act = nn.Sigmoid() 
        
        # Output Projection
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, ts=None): # 🚀 别忘了接收 ts
        # x: (B, L, D)
        B, L, D = x.shape
        
        # 1. Global Path
        # 全局分支没有变形，直接完美传入 ts
        out_g = self.g_ssm(x, ts) 
        
        # 2. Local Path - Fixed Window Partition (Non-overlapping)
        K = self.window_size if self.window_size is not None else max(1, L // 2)
        
        # Calculate padding needed to be divisible by K
        pad_len = (K - L % K) % K
        if pad_len > 0:
            # Pad at the end of the sequence dimension (dim 1)
            x_pad = F.pad(x, (0, 0, 0, pad_len)) 
            
            # =================================================================
            # 🚀 同步 Pad 时间张量 ts
            # ts 形状是 (B, L, 1)，同样在 L 维度（倒数第二维）填充 pad_len 个 0
            if ts is not None:
                ts_pad = F.pad(ts, (0, 0, 0, pad_len))
            # =================================================================
        else:
            x_pad = x
            ts_pad = ts if ts is not None else None
            
        # Partition: (B, L_pad, D) -> (B * N_windows, K, D)
        x_local = einops.rearrange(x_pad, 'b (n k) d -> (b n) k d', k=K)
        
        # =================================================================
        # 🚀 同步 Partition 时间张量 ts
        # (B, L_pad, 1) -> (B * N_windows, K, 1)
        if ts_pad is not None:
            ts_local = einops.rearrange(ts_pad, 'b (n k) c -> (b n) k c', k=K)
        else:
            ts_local = None
        # =================================================================
        
        # Process with L-SSM
        # 🚀 把切好块的局部时间特征传给 l_ssm
        out_l_local = self.l_ssm(x_local, ts_local) 
        
        # Merge: (B * N_windows, K, D) -> (B, L_pad, D)
        out_l_pad = einops.rearrange(out_l_local, '(b n) k d -> b (n k) d', b=B)
        
        # Crop padding
        if pad_len > 0:
            out_l = out_l_pad[:, :L, :]
        else:
            out_l = out_l_pad
            
        # 3. Fusion
        gate = self.gate_act(self.gate_linear(x))
        
        # Gated fusion of Global and Local branches
        out = out_g * gate + out_l * gate 
        
        return self.out_linear(out)


class VMambaTFRModule(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 1, d_state: int = 16, **kwargs):
        super().__init__()
        
        self.d_model = d_model
        # 移除了静态的 temporal_encodings
        
        self.layers = nn.ModuleList([
            SS1D(d_model=d_model, d_state=d_state, **kwargs)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
    def forward(self, features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, D, H, W)
        timestamps: (B, T) 真实的时间戳数据 (例如距离某一天过去的天数)
        """
        B, T, D, H, W = features.size()
        # 1. 动态生成基于时间戳的编码 (B, T, D)
        time_encodings = self.get_timestamp_encodings(timestamps, self.d_model, features.device)
        # 2. Reshape features to tokens: (B * H * W, T, Channels)
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')
        # 3. 将时间编码扩展并相加
        # time_encodings 形状为 (B, T, D)，需要扩展为 (B * H * W, T, D)
        time_encodings = torch.repeat_interleave(time_encodings, H * W, dim=0)
        tokens = tokens + time_encodings
        # 4. Process in chunks
        chunk_size = 4096 
        tokens_out = []
        
        for i in range(0, tokens.size(0), chunk_size):
            chunk = tokens[i:i + chunk_size]
            
            for layer, norm in zip(self.layers, self.norms):
                # Residual connection
                out = layer(chunk)
                chunk = norm(chunk + out)
                
            tokens_out.append(chunk)
        features_hat = torch.cat(tokens_out, dim=0)
        # Reshape to original shape
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)
        return features_hat
    @staticmethod
    def get_timestamp_encodings(timestamps: torch.Tensor, d_model: int, device: torch.device):
        """
        根据连续的时间戳生成正弦/余弦编码
        timestamps: (B, T)
        """
        B, T = timestamps.shape
        pe = torch.zeros(B, T, d_model, device=device)
        
        # 类似于标准 Transformer 的频率项计算
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        
        # 使用真实时间戳乘以频率项
        pe[:, :, 0::2] = torch.sin(timestamps.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(timestamps.unsqueeze(-1) * div_term)
        
        return pe


class GLMambaTFRModule(nn.Module):
    def __init__(self, t: int, d_model: int, n_layers: int = 1, d_state: int = 16, **kwargs):
        super().__init__()
        self.d_model = d_model
        
        # 如果你还想保留默认的固定编码作为备用，可以保留这行
        self.register_buffer('default_temporal_encodings', GLMambaTFRModule.get_relative_encodings(t, d_model), persistent=False)
        
        self.layers = nn.ModuleList([
            GLSSM(d_model=d_model, d_state=d_state, window_size=6, **kwargs)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
    # 辅助函数：根据真实时间戳生成正弦位置编码
    def get_time_encodings(self, timestamps: Tensor) -> Tensor:
        """
        timestamps: (B, T) 真实的连续时间值（例如天数、秒数等）
        返回: (B, T, d_model)
        """
        B, T = timestamps.shape
        pe = torch.zeros(B, T, self.d_model, device=timestamps.device)
        
        # 计算正弦和余弦频率 (类似 Transformer 标准位置编码，但基于真实时间)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=timestamps.device) * 
                             -(math.log(10000.0) / self.d_model))
        
        # timestamps.unsqueeze(-1) 形状变为 (B, T, 1)
        pe[..., 0::2] = torch.sin(timestamps.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(timestamps.unsqueeze(-1) * div_term)
        return pe
    # 修改前向传播，增加 timestamps 参数
    def forward(self, features: Tensor, timestamps: Tensor = None) -> Tensor:
        B, T, D, H, W = features.size()
        # Reshape to tokens: (Batch * Spatial, Time, Channels)
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')
        # 注入时间编码
        #tokens = tokens + self.default_temporal_encodings.to(tokens.device).repeat(B * H * W, 1, 1)
        # =================================================================
        # 🚀 核心新增：时间特征的空间广播与变形
        # =================================================================
        if timestamps is not None:
            # 确保 timestamps 是 (B, T, 1) 的形状
            if timestamps.dim() == 2:
                timestamps = timestamps.unsqueeze(-1)
                
            # 空间广播：由于同一次观测的所有像素共享相同的物理时间差
            # 我们用 expand 而不是 repeat，极其节省显存！
            # (B, T, 1) -> (B, H, W, T, 1)
            ts_expanded = timestamps.unsqueeze(1).unsqueeze(2).expand(B, H, W, T, 1)
            
            # 同样将其展平，形状完全对齐 tokens: (B*H*W, T, 1)
            timestamps_tokens = einops.rearrange(ts_expanded, 'B H W T C -> (B H W) T C')
        else:
            timestamps_tokens = None
        # =================================================================
      # 为了防止显存爆炸的分块处理
        chunk_size = 4096 
        tokens_out = []
        
        for i in range(0, tokens.size(0), chunk_size):
            # 取出当前特征块 (chunk_size, T, D)
            chunk = tokens[i:i + chunk_size]
            
            # 🚀 新增：同步取出对应的时间分块 (chunk_size, T, 1)
            ts_chunk = timestamps_tokens[i:i + chunk_size] if timestamps_tokens is not None else None
            
            for layer, norm in zip(self.layers, self.norms):
                out = layer(chunk, ts_chunk) 
                
                # 残差连接与归一化
                chunk = norm(chunk + out)
                
            tokens_out.append(chunk)
            
        features_hat = torch.cat(tokens_out, dim=0)
        
        # 还原回原来的多维空间形态
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)
        
        return features_hat
    @staticmethod
    def get_relative_encodings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


class TFRModule(nn.Module):
    def __init__(self, t: int, d_model: int, n_heads: int, d_hid: int, activation: str, n_layers: int):
        super().__init__()
        # Generate relative temporal encodings
        self.register_buffer('temporal_encodings', self.get_relative_encodings(t, d_model), persistent=False)

        # Define a transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_hid, batch_first=True,
            activation=activation
        )
        # Create module
        self.temporal_feature_refinement = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, features: Tensor) -> Tensor:
        B, T, D, H, W = features.size()

        # Reshape to tokens
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')

        # Adding relative temporal encodings
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # Feature refinement with self-attention
        features_hat = self.temporal_feature_refinement(tokens)

        # Reshape to original shape
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)

        return features_hat

    @staticmethod
    def get_relative_encodings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result

class CFModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(features: Sequence[Tensor], edges: Sequence[Tuple[int, int]]) -> Sequence[Tensor]:
        # compute urban change detection features
        features_ch = []
        for feature in features:
            B, T, _, H, W = feature.size()
            feature_ch = []
            for t1, t2 in edges:
                feature_ch.append(feature[:, t2] - feature[:, t1])
            # n: number of combinations
            feature_ch = torch.stack(feature_ch)
            features_ch.append(feature_ch)
        return features_ch


class MTIModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, o_ch: Tensor, o_seg: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:

        B, T, _, H, W = o_seg.size()

        # Get processing function by defining the Markov network
        process_pixel = self.markov_network(T, edges)

        # Reshape
        o_ch = einops.rearrange(o_ch, 'B N C H W -> (B H W) N C').cpu().numpy()
        o_seg = einops.rearrange(o_seg, 'B T C H W -> (B H W) T C').cpu().numpy()

        # Find optimal building time series using Markov network
        o_seg_mrf = Parallel(n_jobs=-1)(delayed(process_pixel)(p_seg, p_ch) for p_seg, p_ch, in zip(o_seg, o_ch))

        # Reshape
        o_seg_mrf = torch.Tensor(o_seg_mrf)
        o_seg_mrf = einops.rearrange(o_seg_mrf, '(B H W) (T C) -> B T C H W', H=H, W=W, T=T)

        return o_seg_mrf

    @staticmethod
    def markov_network(n: int, edges: Sequence[Tuple[int, int]]) -> Callable:
        # Define a function to process a single pixel
        def process_pixel(y_hat_seg: Sequence[float], y_hat_ch: Sequence[float]):
            model = MarkovNetwork()

            for t in range(len(y_hat_seg)):
                model.add_node(f'N{t}')
                # Cardinality: number of potential values (i.e., 2: 0/1)
                # Potential Values for node: P(urban=True), P(urban=False)
                urban_value = float(y_hat_seg[t])
                factor = DiscreteFactor([f'N{t}'], cardinality=[2], values=[1 - urban_value, urban_value])
                model.add_factors(factor)

            # add adjacent edges w/o potentials
            for t in range(n - 1):
                model.add_edge(f'N{t}', f'N{t + 1}')

            # add edges with potentials
            for i, (t1, t2) in enumerate(edges):
                model.add_edge(f'N{t1}', f'N{t2}')
                # [P(A=False, B=False), P(A=False, B=True), P(A=True, B=False), P(A=True, B=True)]
                change_value = float(y_hat_ch[i])
                edge_values = [1 - change_value, change_value, change_value, 1 - change_value]

                factor = DiscreteFactor([f'N{t1}', f'N{t2}'], cardinality=[2, 2], values=edge_values)
                model.add_factors(factor)

            # Create an instance of BeliefPropagation algorithm
            bp = BeliefPropagation(model)

            # Compute the most probable state of the MRF
            state = bp.map_query()
            states_list = [state[f'N{t}'] for t in range(n)]
            return states_list

        return process_pixel
