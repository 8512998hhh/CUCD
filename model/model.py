import torch
import torch.nn as nn
from torch import Tensor
import einops
import torch.nn.functional as F

from typing import Tuple, Sequence
from pathlib import Path

from utils.experiment_manager import CfgNode
from torch.utils.checkpoint import checkpoint

from model import unet, modules


class ContUrbanCDModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(ContUrbanCDModel, self).__init__()

        # Attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY

        # ConvNet layers
        self.inc = unet.InConv(self.c, self.topology[0], unet.DoubleConv)
        self.encoder = unet.Encoder(cfg)

        self.decoder_seg = unet.Decoder(cfg)
        self.outc_seg = unet.OutConv(self.topology[0], self.d_out)

        self.decoder_ch = unet.Decoder(cfg)
        self.outc_ch = unet.OutConv(self.topology[0], self.d_out)

        # Temporal feature refinement (TFR) modules
        tfr_modules = []
        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        for i, d_model in enumerate(transformer_dims):
            tfr_module = modules.TFRModule(
                t=self.t,
                d_model=d_model,
                n_heads=cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS,
                d_hid=self.topology[0] * 4,
                activation=cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION,
                n_layers=cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
            )
            tfr_modules.append(tfr_module)
        self.tfr_modules = nn.ModuleList(tfr_modules)
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model, 1, 1))
            for d_model in transformer_dims
        ])

        # Change feature (CF) module
        self.cf_module = modules.CFModule()

        # Multi-task integration (MTI) module
        self.mti_module = modules.MTIModule()

    def forward(self, x: Tensor, edges: Sequence[Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
        B, T, _, H, W = x.size()

        # Feature extraction with Siamese ConvNet encoder
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.encoder(self.inc(x))
        features = [einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B) for f_s in features]

        # Temporal feature refinement with TFR modules
        for i, tfr_module in enumerate(self.tfr_modules):
            f = features[i]  # Feature maps at scale s
            # Feature refinement with self-attention
            f_refined = tfr_module(f)
            features[i] = f_refined

        # Change feature maps
        features_ch = self.cf_module(features, edges)
        features_ch = [einops.rearrange(f, 'n b c h w -> (b n) c h w') for f in features_ch]

        # Building segmentation
        features_seg = [einops.rearrange(f, 'b t c h w -> (b t) c h w') for f in features]
        # 保存最后一层特征供伪标签使用
        self.last_features_seg = self.decoder_seg(features_seg)
        logits_seg = self.outc_seg(self.last_features_seg)
        logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=B)

        logits_ch = self.outc_ch(self.decoder_ch(features_ch))
        logits_ch = einops.rearrange(logits_ch, '(b n) c h w -> b n c h w', n=len(edges))

        return logits_ch, logits_seg

    def inference(self, x: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:
        logits_ch, logits_seg = self.forward(x, edges)
        o_ch = torch.sigmoid(logits_ch).detach()
        o_seg = torch.sigmoid(logits_seg).detach()
        o_seg = self.mti_module(o_ch, o_seg, edges)
        return o_seg


class MaskReconHead(nn.Module):
    """
    轻量级重建头：将解码器输出特征映射回输入像素空间。
    预训练完成后可丢弃，不影响下游 forward/inference。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class ProjectionHead(nn.Module):
    """
    像素级投影头（projection head），仅用于对比学习预训练。

    InfoNCE 直接作用在 backbone/decoder 输出特征上时，会把"对比约束"的扭曲
    烙进下游任务直接消费的表征里，造成负迁移。这里用一个用完即弃的投影头吸收
    这种扭曲：对比损失只作用在 proj 输出上，下游 forward/inference 不经过它，
    微调时可直接丢弃（保留也不影响，因为正常前向不会调用）。

    采用 1x1 卷积实现逐像素 MLP，保持空间分辨率 (H, W) 不变。
    """
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = 128):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class ContUrbanCDMambaModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(ContUrbanCDMambaModel, self).__init__()

        # Attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY

        # ConvNet layers
        self.inc = unet.InConv(self.c, self.topology[0], unet.DoubleConv)
        self.encoder = unet.Encoder(cfg)

        self.decoder_seg = unet.Decoder(cfg)
        self.outc_seg = unet.OutConv(self.topology[0], self.d_out)

        self.decoder_ch = unet.Decoder(cfg)
        self.outc_ch = unet.OutConv(self.topology[0], self.d_out)

        #self.decoder_edge = unet.Decoder(cfg)
        #self.outc_edge = unet.OutConv(self.topology[0], self.d_out)
        # Temporal feature refinement (TFR) modules - USING VMAMBA
        tfr_modules = []
        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        for i, d_model in enumerate(transformer_dims):
            # tfr_module = modules.VMambaTFRModule(
            #     t=self.t,
            #     d_model=d_model,
            #     n_layers=cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
            # )
            tfr_module = modules.GLMambaTFRModule(
                t=self.t,
                d_model=d_model,
                n_layers=cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
            )
                
            tfr_modules.append(tfr_module)
        self.tfr_modules = nn.ModuleList(tfr_modules)
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model, 1, 1))
            for d_model in transformer_dims
        ])

        # Change feature (CF) module
        self.cf_module = modules.CFModule()

        # Multi-task integration (MTI) module
        self.mti_module = modules.MTIModule()

        # 掩码重建头（仅预训练使用，不影响 forward/inference）
        self.recon_head = MaskReconHead(self.topology[0], self.c)

        # 对比学习投影头（仅 CL 预训练使用，下游 forward/inference 不经过它）
        self.proj_head = ProjectionHead(self.topology[0], out_channels=128)
        print(f"Using GLSSM modules")

    def forward_mask_recon(
        self,
        x: Tensor,
        mask: Tensor,
        edges: Sequence[Tuple[int, int]],
        timestamps: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        特征级时间步掩码重建前向传播（仅用于 SSL 预训练）。

        流程：
          1. Encoder 对所有帧正常编码
          2. 在 Encoder 输出的特征图上，将被掩码时间步替换为可学习 mask token
          3. Mamba/TFR 时序模块在掩码特征上做前向传播（被迫从可见帧重建）
          4. Decoder 重建全分辨率特征
          5. 重建头将掩码帧的特征映射回像素空间

        Args:
            x:         (B, T, C_in, H, W) 原始输入
            mask:      (B, T) bool，True = 需要重建的时间步
            edges:     图边（同 forward）
            timestamps: 时间戳（可选）

        Returns:
            pred_pixels:   (N_masked, C_in, H, W) 预测的像素值
            target_pixels: (N_masked, C_in, H, W) 原始像素值（重建目标）
        """
        B, T, C_in, H, W = x.shape

        # 1. 所有帧经 Encoder 提取特征
        x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        inc_x = self.inc(x_flat)
        features = self.encoder(inc_x)
        features_raw = [
            einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B)
            for f_s in features
        ]

        # 2. 特征级掩码：用可学习 token 表示缺失时间步
        for i in range(len(features_raw)):
            mask_exp = mask[:, :, None, None, None].expand_as(features_raw[i])
            token = self.mask_tokens[i].expand_as(features_raw[i])
            features_raw[i] = torch.where(mask_exp, token, features_raw[i])

        # 3. Mamba/TFR 时序模块在掩码特征上传播（被迫从可见帧补全缺失）
        for i, tfr_module in enumerate(self.tfr_modules):
            features_raw[i] = tfr_module(features_raw[i], timestamps)

        # 4. Segmentation Decoder 重建全分辨率特征
        features_seg = [
            einops.rearrange(f, 'b t c h w -> (b t) c h w')
            for f in features_raw
        ]
        last_features = self.decoder_seg(features_seg)          # (B*T, topology[0], H, W)
        last_features = einops.rearrange(
            last_features, '(b t) c h w -> b t c h w', b=B
        )                                                        # (B, T, topology[0], H, W)

        # 5. 仅在掩码位置应用重建头
        pred_feat_masked = last_features[mask]                   # (N_masked, topology[0], H, W)
        pred_pixels = self.recon_head(pred_feat_masked)          # (N_masked, C_in, H, W)

        # 6. 从原始输入取出掩码位置的目标像素
        target_pixels = x[mask]                                  # (N_masked, C_in, H, W)

        return pred_pixels, target_pixels

     # 修改 forward，增加 timestamps 参数
    def forward(self, x: Tensor, edges: Sequence[Tuple[int, int]], timestamps: Tensor = None, return_features: bool = False, project: bool = False):
        B, T, _, H, W = x.size()
        
        # 🚀 修改：时间间隔张量的安全检查与合理的物理 Fallback
        # ==========================================================
        if timestamps is None:
            # 默认假设：均匀采样。除第一帧外，所有相邻帧的间隔为 1。
            # 生成形状为 (B, T, 1) 的全 1 张量
            timestamps = torch.ones(B, T, 1, dtype=torch.float32, device=x.device)
            timestamps[:, 0, :] = 0.0 # 第一帧没有前置帧，间隔为 0
            
        # 防御性代码：如果外部 Dataset 传进来的是 (B, T)，强制补齐到 (B, T, 1)
        elif timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)
        # ==========================================================
            
        # Feature extraction with Siamese ConvNet encoder
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        
        inc_x = self.inc(x)
        # 注意：如果 self.encoder 不支持 kwargs，直接传参即可
        features = self.encoder(inc_x)
        
        features_raw = [einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B) for f_s in features]
        
        # Temporal feature refinement with TFR modules
        for i, tfr_module in enumerate(self.tfr_modules):
            f = features_raw[i]  
            # Mamba 模块本身也可以用 checkpoint，但由于它带有 timestamps 两个参数，
            # 可以写个 wrapper 开启，这里先优化最重的 CNN 部分
            f_refined = tfr_module(f, timestamps)
            features_raw[i] = f_refined
            
        # Change feature maps
        features_ch = self.cf_module(features_raw, edges)
        features_ch = [einops.rearrange(f, 'n b c h w -> (b n) c h w') for f in features_ch]
        
        features_seg = [einops.rearrange(f, 'b t c h w -> (b t) c h w') for f in features_raw]
            
        last_features_seg = self.decoder_seg(features_seg)
        
        if return_features:
            # ==========================================================
            # 🚀 修复与优化：使用解码器输出的高分辨率时序特征做对比学习
            # ==========================================================
            # last_features_seg 的形状是 (B*T, C, H, W)
            if project:
                # 对比学习预训练：先经过投影头，InfoNCE 只作用在投影空间，
                # 避免把对比约束的扭曲直接烙进下游消费的 decoder 特征（缓解负迁移）。
                # proj 输出形状 (B*T, proj_dim, H, W)
                proj_features = self.proj_head(last_features_seg)
                proj_features = einops.rearrange(proj_features, '(b t) c h w -> b t c h w', b=B)
                return proj_features

            # 1. 把它 reshape 回带有时间维度 T 的形状：(B, T, C, H, W)
            features_high_res = einops.rearrange(last_features_seg, '(b t) c h w -> b t c h w', b=B)
            
            # 2. 在时间维度 T 上求均值，消去 T 维度，得到 (B, C, H, W) 
            # 这里的特征既融合了 Mamba 的时序规律，又恢复了 64x64 的高分辨率！
            # features_pool = features_high_res.mean(dim=1) 
            
            return features_high_res
        else:
            logits_seg = self.outc_seg(last_features_seg)
            logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=B)

            
            last_features_ch = self.decoder_ch(features_ch)
        
            logits_ch = self.outc_ch(last_features_ch)
            logits_ch = einops.rearrange(logits_ch, '(b n) c h w -> b n c h w', n=len(edges))
            return logits_ch, logits_seg
    # 同样修改 inference，增加 timestamps 参数并传给 forward
    def inference(self, x: Tensor, edges: Sequence[Tuple[int, int]], timestamps: Tensor = None) -> Tensor:
        logits_ch, logits_seg = self.forward(x, edges, timestamps=timestamps)
        o_ch = torch.sigmoid(logits_ch).detach()
        o_seg = torch.sigmoid(logits_seg).detach()
        o_seg = self.mti_module(o_ch, o_seg, edges)
        return o_seg


def masked_recon_forward(
    net: nn.Module,
    x: Tensor,
    mask: Tensor,
    edges,
    timestamps: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    """
    DataParallel 兼容的掩码重建前向传播入口。
    自动解包 DataParallel 封装，调用 forward_mask_recon。
    """
    core = net.module if isinstance(net, nn.DataParallel) else net
    return core.forward_mask_recon(x, mask, edges, timestamps)


class ContUrbanCDMambaModelCT(ContUrbanCDMambaModel):
    """
    连续时间 (Continuous-Time) 版本模型。

    完全继承 ContUrbanCDMambaModel（encoder / decoder / forward / inference /
    forward_mask_recon 等全部复用），仅把时序精炼模块 tfr_modules 从
    GLMambaTFRModule 替换为 GLMambaTFRModuleCT：时间信息不再通过乘性门控注入，
    而是作为 SSM 连续时间离散化的步长 Δ_eff = softplus(dt_proj(x)+b)·Δt_norm。

    不修改任何现有类，随时可通过关闭 cfg.MODEL.USE_CONTINUOUS_TIME 恢复原模型。
    """
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        from model import modules_ct

        # 时间归一化尺度：可选，默认 1.0（以"月"为单位）。
        ts_scale = getattr(cfg.MODEL, "TS_SCALE", 1.0)
        # local 窗口大小，与原 GLSSM 默认保持一致（6）。
        window_size = getattr(cfg.MODEL, "CT_WINDOW_SIZE", 6)

        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        ct_modules = []
        for d_model in transformer_dims:
            ct_modules.append(modules_ct.GLMambaTFRModuleCT(
                t=self.t,
                d_model=d_model,
                n_layers=cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS,
                window_size=window_size,
                ts_scale=ts_scale,
            ))
        # 替换掉父类构造的 GLMambaTFRModule
        self.tfr_modules = nn.ModuleList(ct_modules)
        print("Using Continuous-Time GLSSM modules (Δ_eff = softplus(dt)·Δt)")


def init_model(cfg: CfgNode) -> nn.Module:
    # Select model based on config or default to original
    # For now, we can manually switch here or add a config parameter
    # Example: if cfg.MODEL.TYPE == "mamba": return ContUrbanCDMambaModel(cfg)

    # Check if a specific config flag is set (this is hypothetical, modify as needed)
    # Defaulting to original to preserve behavior unless explicitly changed
    if hasattr(cfg.MODEL, "USE_MAMBA") and cfg.MODEL.USE_MAMBA:
        # 🚀 连续时间开关：默认关闭，开启后使用 CT 时序模块，其余结构不变
        if getattr(cfg.MODEL, "USE_CONTINUOUS_TIME", False):
            net = ContUrbanCDMambaModelCT(cfg)
        else:
            net = ContUrbanCDMambaModel(cfg)
    else:
        net = ContUrbanCDModel(cfg)

    return torch.nn.DataParallel(net)


def save_model(network: nn.Module, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'weights': network.state_dict(),
    }
    torch.save(checkpoint, save_file)

def save_model_pretrained(network: nn.Module, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / "pretrained.pt"
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'weights': network.state_dict(),
    }
    torch.save(checkpoint, save_file)

def save_model_latest(network: nn.Module, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}_latest.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'weights': network.state_dict(),
    }
    torch.save(checkpoint, save_file)

def _resolve_state_dict_key(key: str, model_keys: set) -> str:
    """将 checkpoint 中的 key 映射到当前 model.state_dict() 中存在的 key。"""
    if key in model_keys:
        return key
    if key.startswith('module.') and key[len('module.'):] in model_keys:
        return key[len('module.'):]
    prefixed = f'module.{key}'
    if prefixed in model_keys:
        return prefixed
    return key


def _top_level_module(key: str) -> str:
    name = key[len('module.'):] if key.startswith('module.') else key
    return name.split('.')[0]


def load_state_dict_compatible(net: nn.Module, state_dict: dict):
    """
    只加载 checkpoint 中与当前模型同名且 shape 一致的参数。
    checkpoint 中缺失的模块保持随机初始化，并打印提示。
    """
    model_sd = net.state_dict()
    model_keys = set(model_sd.keys())

    filtered = {}
    skipped_shape = []
    unused_ckpt_keys = []

    for ckpt_key, ckpt_val in state_dict.items():
        model_key = _resolve_state_dict_key(ckpt_key, model_keys)
        if model_key not in model_keys:
            unused_ckpt_keys.append(ckpt_key)
            continue
        if model_sd[model_key].shape != ckpt_val.shape:
            skipped_shape.append(
                f'{ckpt_key} (ckpt {tuple(ckpt_val.shape)} vs model {tuple(model_sd[model_key].shape)})'
            )
            continue
        filtered[model_key] = ckpt_val

    incompatible = net.load_state_dict(filtered, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print(f'[Load] Loaded {len(filtered)}/{len(model_sd)} model tensors from checkpoint.')

    if skipped_shape:
        print(f'[Load] Skipped {len(skipped_shape)} tensors due to shape mismatch:')
        for item in skipped_shape[:10]:
            print(f'  - {item}')
        if len(skipped_shape) > 10:
            print(f'  ... and {len(skipped_shape) - 10} more')

    if missing_keys:
        missing_by_module = {}
        for key in missing_keys:
            mod = _top_level_module(key)
            missing_by_module.setdefault(mod, []).append(key)
        print(
            f'[Load] {len(missing_keys)} parameters not found in checkpoint '
            f'(using random initialization):'
        )
        for mod, keys in sorted(missing_by_module.items()):
            print(f'  - {mod}: {len(keys)} param(s)')

    if unused_ckpt_keys:
        print(f'[Load] {len(unused_ckpt_keys)} checkpoint keys have no match in current model (ignored).')
        for key in unused_ckpt_keys[:10]:
            print(f'  - {key}')
        if len(unused_ckpt_keys) > 10:
            print(f'  ... and {len(unused_ckpt_keys) - 10} more')

    if unexpected_keys:
        print(f'[Load] Warning: {len(unexpected_keys)} unexpected keys after load (should be empty).')

    return incompatible


# 仅迁移这些模块的预训练权重：输入卷积 stem + Encoder + Mamba 时序模块。
# 其余模块（decoder_seg / decoder_ch / outc_* / proj_head / recon_head 等）
# 一律从零开始训练，避免把 CL 的时间平滑先验带入下游解码器。
PRETRAIN_TRANSFER_PREFIXES = ('inc', 'encoder', 'tfr_modules')


def load_pretrained_encoder_mamba(net: nn.Module, state_dict: dict,
                                  module_prefixes: Sequence[str] = PRETRAIN_TRANSFER_PREFIXES):
    """
    只加载 Encoder 与 Mamba 时序模块（以及输入卷积 stem）的预训练权重。

    checkpoint 中其它模块的权重会被丢弃，下游对应模块保持随机初始化。
    兼容 DataParallel 的 'module.' 前缀，并按张量形状二次校验，避免误加载。
    """
    model_sd = net.state_dict()

    def _strip(name: str) -> str:
        return name[len('module.'):] if name.startswith('module.') else name

    filtered = {}
    skipped_shape = []
    for k, v in state_dict.items():
        name = _strip(k)
        if not any(name == p or name.startswith(p + '.') for p in module_prefixes):
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped_shape.append(k)

    incompatible = net.load_state_dict(filtered, strict=False)
    unexpected_keys = list(incompatible.unexpected_keys)

    print(f"[Pretrain] Selective transfer for modules {module_prefixes}: "
          f"loaded {len(filtered)} tensors.")
    if skipped_shape:
        print(f"[Pretrain] Skipped {len(skipped_shape)} tensors due to shape mismatch.")
    if unexpected_keys:
        print(f"[Pretrain] Warning: {len(unexpected_keys)} unexpected keys ignored.")
    return incompatible

def load_model(cfg: CfgNode, device: torch.device, pretrained: bool = False) -> nn.Module:
    net = init_model(cfg)
    net.to(device)
    if pretrained:
        net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / 'pretrained.pt'
        checkpoint = torch.load(net_file, map_location=device)
        load_pretrained_encoder_mamba(net, checkpoint['weights'])
    else:
        net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
        checkpoint = torch.load(net_file, map_location=device)
        load_state_dict_compatible(net, checkpoint['weights'])
    #net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / 'conturbancd_sn7_GLMamba.pt'
    print("load model from: ", net_file)
    return net

def load_model_IL(cfg: CfgNode, device: torch.device) -> nn.Module:
    net = init_model(cfg)
    net.to(device)
    net_file = cfg.MODEL.PRETRAINED_PATH
    checkpoint = torch.load(net_file, map_location=device)
    load_state_dict_compatible(net, checkpoint['weights'])
    return net


def power_jaccard_loss(input: Tensor, target: Tensor, disable_sigmoid: bool = False) -> Tensor:
    input_sigmoid = torch.sigmoid(input) if not disable_sigmoid else input
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)

def point_power_jaccard_loss(input: Tensor, target: Tensor, loss_mask: Tensor, disable_sigmoid: bool = False) -> Tensor:
    """
    用于点监督的 Power Jaccard Loss。
    
    Args:
        input: 模型预测输出，形状为 (B, T, 1, H, W)
        target: 点监督的真实标签 (y_point)，形状为 (B, T, 1, H, W)
        loss_mask: 损失掩码，形状为 (B, 1, H, W)，值为1表示该点参与损失计算，0表示忽略
        disable_sigmoid: 是否禁用 sigmoid 激活
        
    Returns:
        Tensor: 计算得到的标量损失值
    """
    input_sigmoid = torch.sigmoid(input) if not disable_sigmoid else input
    eps = 1e-6
    
    # 将 loss_mask 在时间维度上广播，使其形状与 input 和 target 匹配
    # mask 形状从 (B, 1, H, W) 变为 (B, T, 1, H, W)
    # 注意：如果 loss_mask 本身已经是 (B, T, 1, H, W) 也没关系，unsqueeze 会自动处理，
    # 但根据我们在数据集中的定义，它是 (B, 1, H, W)，所以需要扩展时间维度 T
    # target 的形状是 (B, T, 1, H, W)
    # print(loss_mask.shape)
    # print(input.shape)
    # print(loss_mask.sum())
    if loss_mask.dim() == 4 and input.dim() == 5:
        # loss_mask: (B, 1, H, W) -> (B, 1, 1, H, W) 然后在 T 维度广播
        mask = loss_mask.unsqueeze(1).expand_as(input)
    elif loss_mask.dim() == 3 and input.dim() == 4:
        # 兼容没有 T 维度的情况 (B, 1, H, W)
        mask = loss_mask.expand_as(input)
    else:
        mask = loss_mask

    # 仅保留被 mask 选中的点
    iflat = input_sigmoid[mask > 0.5]
    tflat = target[mask > 0.5]
    
    # 如果没有点被选中（极少数情况），返回 0 损失并保持梯度连通性
    if iflat.numel() == 0:
        return (input_sigmoid * 0).sum()

    intersection = (iflat * tflat).sum()
    denom = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)

def logical_consistency_loss_fast(pred_seg, pred_ch, edges):
    """
    加速版的逻辑一致性损失。
    通过张量高级索引 (Advanced Indexing) 避免 Python 层的 for 循环和 torch.stack，
    直接在 GPU 上进行并行的张量运算。
    
    pred_seg: 建筑分割概率输出 (经过Sigmoid), 形状 [B, T, 1, H, W]
    pred_ch: 变化检测概率输出 (经过Sigmoid), 形状 [B, E, 1, H, W]
    edges: 时间步组合列表，形如 [(0, 1), (1, 2), ...]
    """
    # 将 edges 转换为 tensor，形状为 [E, 2]
    # 注意：为了避免每次调用都把 edges 搬到 GPU，这里可以只在 CPU 提取索引
    t1_indices = [e[0] for e in edges]
    t2_indices = [e[1] for e in edges]
    
    # 利用 PyTorch 的高级索引，一次性提取所有 t1 和 t2 时刻的特征
    # S_t1 和 S_t2 的形状将变为 [B, E, 1, H, W]
    S_t1 = pred_seg[:, t1_indices]
    S_t2 = pred_seg[:, t2_indices]
    
    # 向量化计算软逻辑 XOR
    # 所有的 E 条边会同时在 GPU 上并行计算
    expected_change = S_t2 * (1 - S_t1) + S_t1 * (1 - S_t2)
    
    # 计算 MSE 损失
    loss_logic = F.mse_loss(pred_ch, expected_change)
    
    return loss_logic

def binary_edge_loss(logits_edge, target_edge):
    """
    专门针对“极度不平衡的二值边缘图”设计的组合损失函数
    
    logits_edge: 模型输出的边缘预测 (未经过 sigmoid!), 形状 [B, T, 1, H, W] 或 [B, 1, H, W]
    target_edge: 真实的二值边缘标签 (只有 0 和 1), 形状需要与 logits 匹配
    """
    # 确保维度对齐
    if logits_edge.dim() == 5 and target_edge.dim() == 4:
        target_edge = target_edge.unsqueeze(1).expand_as(logits_edge)
        
    # ==========================================
    # 1. 动态加权的 BCE Loss (BCEWithLogitsLoss)
    # ==========================================
    # 统计正样本(边缘)和负样本(背景)的数量
    num_pos = torch.sum(target_edge == 1.0)
    num_neg = torch.sum(target_edge == 0.0)
    
    # 动态计算 pos_weight：负样本数量 / 正样本数量
    # 比如全图只有 2% 是边缘，pos_weight 就会是 49。
    # clamp 限制最大值，防止某一个 patch 里完全没有边缘导致无穷大或梯度爆炸
    weight = torch.clamp(num_neg / (num_pos + 1e-6), max=50.0)
    
    # 计算带权重的 BCE (使用 logits 更稳定)
    loss_bce = F.binary_cross_entropy_with_logits(logits_edge, target_edge, pos_weight=weight)
    
    # # ==========================================
    # # 2. 软 Dice Loss (类似你之前的 Power Jaccard)
    # # ==========================================
    # pred_prob = torch.sigmoid(logits_edge)
    # smooth = 1e-6
    
    # # 计算交集和分母
    # intersection = (pred_prob * target_edge).sum()
    # denom = pred_prob.sum() + target_edge.sum()
    
    # # Dice 公式: 1 - (2 * 交集) / (预测和 + 真实和)
    # loss_dice = 1.0 - ((2.0 * intersection + smooth) / (denom + smooth))
    
    # ==========================================
    # 3. 组合返回
    # ==========================================
    # 权重可以根据实验微调，通常 BCE 是主力，Dice 起辅助连贯作用
    return loss_bce

def generate_pseudo_labels_by_confidence(logits_seg, y_point, loss_mask, conf_thresh=0.95):
    # 加上 torch.no_grad() 彻底保护
    with torch.no_grad():
        # 必须使用 .detach() 切断梯度！
        prob_seg = torch.sigmoid(logits_seg.detach())
        
        pseudo_mask = torch.zeros_like(loss_mask)
        pseudo_target = torch.zeros_like(y_point)
        
        is_pseudo_build = (prob_seg > conf_thresh) & (loss_mask == 0)
        is_pseudo_bg = (prob_seg < (1.0 - conf_thresh)) & (loss_mask == 0)
        
        pseudo_mask[is_pseudo_build] = 1.0
        pseudo_mask[is_pseudo_bg] = 1.0
        pseudo_target[is_pseudo_build] = 1.0
        pseudo_target[is_pseudo_bg] = 0.0
        
    return pseudo_mask, pseudo_target

# =====================================================================
# 🚀 新增：专为 64x64 小图序列设计的像素级 InfoNCE 损失
# =====================================================================
def info_nce_loss_pixel_sample(features_full, features_masked, temperature=0.1, num_samples=2048):
    """
    计算基于时空位置采样的 InfoNCE 损失。
    支持两种输入形状：
      - (B, C, H, W)      : 无时间维度（空间模式）
      - (B, T, C, H, W)   : 保留时间维度（时空模式，推荐）
    正样本对定义：同一时空位置 (b, [t,] h, w) 的 Student 特征 vs Teacher 特征。
    负样本对定义：不同时空位置的 Teacher 特征。
    """
    if features_full.dim() == 5:
        B, T, C, H, W = features_full.shape
        # (B, T, C, H, W) → (B*T*H*W, C)，每个时空位置作为独立样本点
        # 对于 B=12, T=12, H=W=64，共 589824 个样本点
        f_full_flat = features_full.permute(0, 1, 3, 4, 2).reshape(-1, C)
        f_mask_flat = features_masked.permute(0, 1, 3, 4, 2).reshape(-1, C)
    else:
        B, C, H, W = features_full.shape
        # (B, C, H, W) → (B*H*W, C)
        f_full_flat = features_full.permute(0, 2, 3, 1).reshape(-1, C)
        f_mask_flat = features_masked.permute(0, 2, 3, 1).reshape(-1, C)

    # 随机抽取 num_samples 个时空位置
    total_tokens = f_full_flat.shape[0]
    num_samples = min(num_samples, total_tokens)
    indices = torch.randperm(total_tokens, device=features_full.device)[:num_samples]

    # 同一组 indices 同时索引 Student(anchor) 和 Teacher(positive)，保证正样本对位置对齐
    anchor   = F.normalize(f_mask_flat[indices], dim=-1)  # (num_samples, C)
    positive = F.normalize(f_full_flat[indices], dim=-1)  # (num_samples, C)

    # 相似度矩阵 (num_samples, num_samples)，对角线为正样本
    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(num_samples, device=anchor.device)

    return F.cross_entropy(logits, labels)


# =====================================================================
# 时间增强专用损失：空间全局平均池化，用完整时序序列做对比
# =====================================================================
def info_nce_loss_temporal_seq(features_full, features_masked, temperature=0.1):
    """
    用于时间高频互换增强的 InfoNCE 损失。
    - 空间维度 (H, W) 做全局平均池化，消去空间细节，保留时序结构。
    - 每个样本单元 = 一条完整时序序列 (T*C 维向量)，共 B 个样本。
    - 正样本对：同一图像的原始序列 vs 时间增强序列。
    - 负样本对：不同图像的原始序列。

    输入形状: (B, T, C, H, W)
    """
    B, T, C, H, W = features_full.shape

    # 空间全局平均池化: (B, T, C, H, W) → (B, T, C)
    f_full = features_full.mean(dim=(-2, -1))
    f_mask = features_masked.mean(dim=(-2, -1))

    # 展平时序: (B, T*C)
    anchor   = F.normalize(f_mask.reshape(B, T * C), dim=-1)
    positive = F.normalize(f_full.reshape(B, T * C), dim=-1)

    # B×B 相似度矩阵，对角线为正样本
    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(B, device=anchor.device)

    return F.cross_entropy(logits, labels)


# =====================================================================
# 空间增强专用损失：时间全局平均池化，用完整空间图做对比
# =====================================================================
def info_nce_loss_spatial_seq(features_full, features_masked, temperature=0.1):
    """
    用于空间低频互换增强的 InfoNCE 损失。
    - 时间维度 (T) 做全局平均池化，消去时序变化，保留空间结构。
    - 每个样本单元 = 一张完整空间特征图 (C*H*W 维向量)，共 B 个样本。
    - 正样本对：同一图像的原始空间图 vs 空间增强空间图。
    - 负样本对：不同图像的原始空间图。

    输入形状: (B, T, C, H, W)
    """
    B, T, C, H, W = features_full.shape

    # 时间全局平均池化: (B, T, C, H, W) → (B, C, H, W)
    f_full = features_full.mean(dim=1)
    f_mask = features_masked.mean(dim=1)

    # 展平空间图: (B, C*H*W)
    anchor   = F.normalize(f_mask.reshape(B, C * H * W), dim=-1)
    positive = F.normalize(f_full.reshape(B, C * H * W), dim=-1)

    # B×B 相似度矩阵，对角线为正样本
    logits = torch.matmul(anchor, positive.T) / temperature
    labels = torch.arange(B, device=anchor.device)

    return F.cross_entropy(logits, labels)


def info_nce_loss_dense_adjacent(features, num_samples=512, temperature=0.1):
    """
    保留空间维度的像素级时序对比损失。
    - 正样本：同一空间坐标下，严格相邻的两帧 (T_i 和 T_{i+1})。
    - 负样本：同一相邻帧对内，其他 Batch 图像中的空间坐标点。
      注意：同一图像内的其他像素不作为负样本，降低同类像素被错误推远的风险；
      不同时间对之间也不互作负样本，避免把同一位置的其他时间帧错误推远。
    - 内存优化：使用空间随机采样，并逐时间对计算相似度矩阵。
    
    输入形状: (B, T, C, H, W)
    """
    B, T, C, H, W = features.shape
    if T < 2:
        return features.new_tensor(0.0)

    # ---------------------------------------------------------
    # 1. 空间随机采样 (解决显存爆炸的核心)
    # ---------------------------------------------------------
    feat = features.reshape(B, T, C, -1) # 形状: (B, T, C, H*W)
    
    # 在 H*W 个像素中，随机抽取 num_samples 个点
    num_samples = min(num_samples, H * W)
    perm = torch.randperm(H * W, device=features.device)[:num_samples]
    feat_sampled = feat[:, :, :, perm] # 形状: (B, T, C, K) 此时 K = num_samples

    # ---------------------------------------------------------
    # 2. 逐相邻帧对计算 InfoNCE
    # ---------------------------------------------------------
    losses = []
    labels = torch.arange(B * num_samples, device=features.device)
    # 方案 A：屏蔽同一图像内部的所有非正样本 logits。
    # 展平顺序为 (b, spatial_idx)，因此第 i 个 token 的 batch id 是 i // num_samples。
    # mask 为 True 的位置会被置为 -inf，不参与 softmax 分母。
    token_batch_ids = torch.arange(B, device=features.device).repeat_interleave(num_samples)
    same_image = token_batch_ids[:, None] == token_batch_ids[None, :]
    positive = torch.eye(B * num_samples, dtype=torch.bool, device=features.device)
    same_image_non_positive = same_image & ~positive

    for t in range(T - 1):
        # Anchor: 第 t 帧；Target: 第 t+1 帧。
        # 展平后每个样本单元是一个 (batch_id, spatial_idx) 点。
        anchor = feat_sampled[:, t].permute(0, 2, 1).reshape(-1, C)      # (B*K, C)
        target = feat_sampled[:, t + 1].permute(0, 2, 1).reshape(-1, C)  # (B*K, C)

        anchor = F.normalize(anchor, dim=-1)
        target = F.normalize(target, dim=-1)

        # 相似度矩阵只在当前相邻时间对内部构造：
        # 对角线是同一 batch、同一空间点的正样本；
        # 非对角线只保留其它 batch 图像的空间点作为负样本。
        sim = torch.matmul(anchor, target.T) / temperature
        sim = sim.masked_fill(same_image_non_positive, float('-inf'))
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_centered = sim - sim_max.detach()
        losses.append(F.cross_entropy(sim_centered, labels))

    return torch.stack(losses).mean()


# =====================================================================
# TiDEC: Time-Disentangled Edge-aware Dense Contrastive Learning
#   L_aug    : 同帧双增强视图，同坐标点拉近（光照/颜色不变）
#   L_stable : 仅对“时序稳定点”拉近参考帧与最远帧（无建筑 mask）
# =====================================================================

def color_augment_batch(x: Tensor, brightness: float = 0.2, contrast: float = 0.2) -> Tensor:
    """对 (B, T, C, H, W) 图像做轻量颜色增强，几何位置不变。"""
    b_factor = 1.0 + (torch.rand(x.size(0), x.size(1), 1, 1, 1, device=x.device) * 2 - 1) * brightness
    c_factor = 1.0 + (torch.rand(x.size(0), x.size(1), 1, 1, 1, device=x.device) * 2 - 1) * contrast
    mean = x.mean(dim=(3, 4), keepdim=True)
    out = (x - mean) * c_factor + mean
    out = out * b_factor
    return out.clamp(0.0, 1.0)


def sample_balanced_point_indices(
    x: Tensor,
    num_samples: int,
    high_ratio: float = 0.3,
    low_ratio: float = 0.3,
) -> Tensor:
    """
    无 mask 的均衡点采样：高梯度 / 低梯度 / 随机。
    返回 (B, K) 的 flatten 空间索引。
    """
    B, _, _, H, W = x.shape
    num_pixels = H * W
    num_samples = min(num_samples, num_pixels)
    n_high = int(num_samples * high_ratio)
    n_low = int(num_samples * low_ratio)
    n_rand = max(0, num_samples - n_high - n_low)

    gray = x[:, 0].mean(dim=1)  # (B, H, W)，用首帧
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    grad = torch.zeros_like(gray)
    grad[:, :, 1:] += gx.abs()
    grad[:, :, :-1] += gx.abs()
    grad[:, 1:, :] += gy.abs()
    grad[:, :-1, :] += gy.abs()

    flat_grad = grad.reshape(B, -1)
    indices = []
    for b in range(B):
        g = flat_grad[b]
        high_idx = torch.topk(g, min(n_high, num_pixels)).indices if n_high > 0 else g.new_zeros(0, dtype=torch.long)
        low_idx = torch.topk(-g, min(n_low, num_pixels)).indices if n_low > 0 else g.new_zeros(0, dtype=torch.long)
        rand_idx = torch.randperm(num_pixels, device=x.device)[:n_rand] if n_rand > 0 else g.new_zeros(0, dtype=torch.long)
        idx = torch.cat([high_idx, low_idx, rand_idx])
        if idx.numel() < num_samples:
            pad = torch.randperm(num_pixels, device=x.device)[: num_samples - idx.numel()]
            idx = torch.cat([idx, pad])
        indices.append(idx[:num_samples])
    return torch.stack(indices, dim=0)


def gather_point_features(features: Tensor, point_idx: Tensor) -> Tensor:
    """features: (B, T, C, H, W), point_idx: (B, K) -> (B, T, C, K)"""
    B, T, C, H, W = features.shape
    K = point_idx.shape[1]
    feat_flat = features.reshape(B, T, C, -1)
    idx = point_idx.view(B, 1, 1, K).expand(B, T, C, K)
    return torch.gather(feat_flat, 3, idx)


def dense_aug_consistency_loss(
    f_aug1: Tensor,
    f_aug2: Tensor,
    point_idx: Tensor,
    temperature: float = 0.1,
    stop_grad_target: bool = True,
) -> Tensor:
    """
    同帧双增强的像素级 InfoNCE。

    正样本：同一图像、同一时间、同一空间坐标的两种颜色增强视图。
    负样本：同一时间下，其它 batch 图像中的采样空间点。

    不使用同一图像内其它像素作为负样本，避免建筑/背景等同类像素被错误推远；
    也不把其它时间帧混作负样本，避免把同一位置的时序相关特征错误推远。
    """
    g1 = gather_point_features(f_aug1, point_idx)
    g2 = gather_point_features(f_aug2, point_idx)
    B, T, C, K = g1.shape

    if stop_grad_target:
        g2 = g2.detach()

    labels = torch.arange(B * K, device=f_aug1.device)
    token_batch_ids = torch.arange(B, device=f_aug1.device).repeat_interleave(K)
    same_image = token_batch_ids[:, None] == token_batch_ids[None, :]
    positive = torch.eye(B * K, dtype=torch.bool, device=f_aug1.device)
    same_image_non_positive = same_image & ~positive

    losses = []
    for t in range(T):
        anchor = g1[:, t].permute(0, 2, 1).reshape(-1, C)
        target = g2[:, t].permute(0, 2, 1).reshape(-1, C)

        anchor = F.normalize(anchor, dim=-1)
        target = F.normalize(target, dim=-1)

        logits = torch.matmul(anchor, target.T) / temperature
        logits = logits.masked_fill(same_image_non_positive, float('-inf'))
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        losses.append(F.cross_entropy(logits, labels))

    return torch.stack(losses).mean()


def dense_stable_temporal_loss(
    features: Tensor,
    point_idx: Tensor,
    stable_ratio: float = 0.5,
    t_ref: int = 0,
    t_short: int = 1,
    t_long: int = -1,
) -> Tensor:
    """
    CACo 思想的 dense 版：仅对稳定点拉近参考帧与最远帧。
    稳定判定：ratio = d_long / d_short，取 ratio 最小的 stable_ratio 比例点。
    """
    T = features.shape[1]
    if T < 2:
        return features.new_tensor(0.0)

    t_long = T - 1 if t_long < 0 else min(t_long, T - 1)
    t_short = min(t_short, T - 1)

    with torch.no_grad():
        g = gather_point_features(features.detach(), point_idx)
        f0 = F.normalize(g[:, t_ref], dim=1)
        fs = F.normalize(g[:, t_short], dim=1)
        fl = F.normalize(g[:, t_long], dim=1)
        d_short = (f0 - fs).pow(2).sum(dim=1).sqrt()
        d_long = (f0 - fl).pow(2).sum(dim=1).sqrt()
        ratio = d_long / (d_short + 1e-6)
        thresh = torch.quantile(ratio, stable_ratio, dim=1, keepdim=True)
        stable = ratio <= thresh

    g = gather_point_features(features, point_idx)
    f0 = F.normalize(g[:, t_ref], dim=1)
    fl = F.normalize(g[:, t_long], dim=1)
    cos = (f0 * fl).sum(dim=1)
    if stable.sum() == 0:
        return features.new_tensor(0.0)
    return (1.0 - cos)[stable].mean()


def tidex_contrastive_loss(
    features: Tensor,
    f_aug1: Tensor,
    f_aug2: Tensor,
    x: Tensor,
    num_samples: int = 256,
    lambda_aug: float = 1.0,
    lambda_stable: float = 1.0,
    stable_ratio: float = 0.5,
) -> Tuple[Tensor, dict]:
    """
    TiDEC 总损失 = lambda_aug * L_aug + lambda_stable * L_stable
    """
    point_idx = sample_balanced_point_indices(x, num_samples=num_samples)
    loss_aug = dense_aug_consistency_loss(f_aug1, f_aug2, point_idx)
    loss_stable = dense_stable_temporal_loss(
        features, point_idx, stable_ratio=stable_ratio,
    )
    loss = lambda_aug * loss_aug + lambda_stable * loss_stable
    stats = {
        'loss_aug': loss_aug.detach(),
        'loss_stable': loss_stable.detach(),
    }
    return loss, stats