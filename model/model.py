import torch
import torch.nn as nn
from torch import Tensor
import einops
import torch.nn.functional as F

from typing import Tuple, Sequence
from pathlib import Path

from utils.experiment_manager import CfgNode

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

        # Change feature (CF) module
        self.cf_module = modules.CFModule()

        # Multi-task integration (MTI) module
        self.mti_module = modules.MTIModule()
        print(f"Using GLSSM modules")

     # 修改 forward，增加 timestamps 参数
    def forward(self, x: Tensor, edges: Sequence[Tuple[int, int]], timestamps: Tensor = None, return_features: bool = False) -> Tuple[Tensor, Tensor]:
        B, T, _, H, W = x.size()
        
        # ==========================================================
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
        features = self.encoder(self.inc(x))
        features_raw = [einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B) for f_s in features]
        # features_smooth = []
        # Temporal feature refinement with TFR modules
        for i, tfr_module in enumerate(self.tfr_modules):
            f = features_raw[i]  # Feature maps at scale s
            # Feature refinement with Mamba/SSM
            # 将 timestamps 传给 TFR 模块
            f_refined = tfr_module(f, timestamps)
            features_raw[i] = f_refined
        # Change feature maps
        #features_for_ch = [f_raw + f_smooth for f_raw, f_smooth in zip(features_raw, features_smooth)]
        features_ch = self.cf_module(features_raw, edges)
        features_ch = [einops.rearrange(f, 'n b c h w -> (b n) c h w') for f in features_ch]
        
        # Building segmentation
        features_seg = [einops.rearrange(f, 'b t c h w -> (b t) c h w') for f in features_raw]
        # 保存最后一层特征供伪标签使用
        last_features_seg = self.decoder_seg(features_seg)
        
        logits_seg = self.outc_seg(last_features_seg)
        logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=B)
        
        logits_ch = self.outc_ch(self.decoder_ch(features_ch))
        logits_ch = einops.rearrange(logits_ch, '(b n) c h w -> b n c h w', n=len(edges))
        
        if return_features:
            # ==========================================================
            # 🚀 修复与优化：使用解码器输出的高分辨率时序特征做对比学习
            # ==========================================================
            # last_features_seg 的形状是 (B*T, C, H, W)
            # 1. 把它 reshape 回带有时间维度 T 的形状：(B, T, C, H, W)
            features_high_res = einops.rearrange(last_features_seg, '(b t) c h w -> b t c h w', b=B)
            
            # 2. 在时间维度 T 上求均值，消去 T 维度，得到 (B, C, H, W) 
            # 这里的特征既融合了 Mamba 的时序规律，又恢复了 64x64 的高分辨率！
            features_pool = features_high_res.mean(dim=1) 
            
            return logits_ch, logits_seg, features_pool
        else:
            return logits_ch, logits_seg
    # 同样修改 inference，增加 timestamps 参数并传给 forward
    def inference(self, x: Tensor, edges: Sequence[Tuple[int, int]], timestamps: Tensor = None) -> Tensor:
        logits_ch, logits_seg = self.forward(x, edges, timestamps=timestamps)
        o_ch = torch.sigmoid(logits_ch).detach()
        o_seg = torch.sigmoid(logits_seg).detach()
        o_seg = self.mti_module(o_ch, o_seg, edges)
        return o_seg


def init_model(cfg: CfgNode) -> nn.Module:
    # Select model based on config or default to original
    # For now, we can manually switch here or add a config parameter
    # Example: if cfg.MODEL.TYPE == "mamba": return ContUrbanCDMambaModel(cfg)
    
    # Check if a specific config flag is set (this is hypothetical, modify as needed)
    # Defaulting to original to preserve behavior unless explicitly changed
    if hasattr(cfg.MODEL, "USE_MAMBA") and cfg.MODEL.USE_MAMBA:
        net = ContUrbanCDMambaModel(cfg)
    else:
        net = ContUrbanCDModel(cfg)
        
    return torch.nn.DataParallel(net)

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

def load_model_prtrained(cfg: CfgNode, device: torch.device) -> nn.Module:
    net = init_model(cfg)
    net.to(device)
    net_file = '/root/autodl-tmp/ContUrbanCD-master/pretrained_TiDEC.pt'
    #net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / 'conturbancd_sn7_GLMamba.pt'
    print("load model from: ", net_file)
    checkpoint = torch.load(net_file, map_location=device)
    load_pretrained_encoder_mamba(net, checkpoint['weights'])
    print(f"load model from {net_file}")
    return net


def save_model(network: nn.Module, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
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

def load_model(cfg: CfgNode, device: torch.device) -> nn.Module:
    net = init_model(cfg)
    net.to(device)
    net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
    #net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / 'conturbancd_sn7_GLMamba.pt'
    print("load model from: ", net_file)
    checkpoint = torch.load(net_file, map_location=device)
    net.load_state_dict(checkpoint['weights'])
    return net

def load_model_IL(cfg: CfgNode, device: torch.device) -> nn.Module:
    net = init_model(cfg)
    net.to(device)
    net_file = cfg.MODEL.PRETRAINED_PATH
    checkpoint = torch.load(net_file, map_location=device)
    net.load_state_dict(checkpoint['weights'])
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
    计算基于像素采样的 InfoNCE 损失。
    features_full: (B, C_out, H, W) - 完整特征 (Teacher)
    features_masked: (B, C_out, H, W) - 掩码特征 (Student)
    num_samples: 每次采样的点数，建议 2048 (权衡显存与负样本数量)
    """
    B, C, H, W = features_full.shape
    
    # 1. 展平空间维度: 把所有 Batch 的所有像素全部混在一起 (B*H*W, C)
    # 对于 B=12, 64x64, 这里会有 49152 个特征点
    f_full_flat = features_full.permute(0, 2, 3, 1).reshape(-1, C)
    f_mask_flat = features_masked.permute(0, 2, 3, 1).reshape(-1, C)

    # 2. 随机抽取 num_samples 个点
    total_pixels = f_full_flat.shape[0]
    num_samples = min(num_samples, total_pixels)
    
    # 随机打乱索引并截取前 num_samples 个
    indices = torch.randperm(total_pixels, device=features_full.device)[:num_samples]

    # 3. 提取对应的 Anchor(掩码特征) 和 Positive(完整特征)
    anchor = f_mask_flat[indices]   # (num_samples, C)
    positive = f_full_flat[indices] # (num_samples, C)

    # 4. L2 归一化
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # 5. 计算相似度矩阵 (num_samples, num_samples)
    logits = torch.matmul(anchor, positive.T) / temperature
    
    # 6. 对角线上的元素是正样本，其他都是负样本
    labels = torch.arange(num_samples, device=anchor.device)
    
    return F.cross_entropy(logits, labels)
# =====================================================================