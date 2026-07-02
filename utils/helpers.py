import torch
from torch import Tensor
from itertools import combinations
from pathlib import Path
from utils.experiment_manager import CfgNode
import json
from typing import Sequence, Tuple
import torch.nn.functional as F

def get_aoi_ids(cfg: CfgNode) -> Sequence[str]:
    aoi_ids = list(cfg.DATASET.TRAIN_IDS)
    return aoi_ids


def get_edges(n: int, edge_type: str) -> Sequence[Tuple[int, int]]:
    """
    根据序列长度 n 动态生成时间戳组合 (Edges)。
    """
    # 🛡️ 终极护栏：变化检测至少需要 2 帧图像！
    if n < 2:
        raise ValueError(f"生成 edges 失败：序列长度 n={n} 小于 2，无法进行变化检测！")

    if edge_type == 'adjacent':
        # 相邻变化：0->1, 1->2, 2->3
        edges = [(t1, t1 + 1) for t1 in range(n - 1)]
        
    elif edge_type == 'cyclic':
        # 环形变化：相邻变化 + 首尾闭环
        edges = [(t1, t1 + 1) for t1 in range(n - 1)]
        # 防止 n=2 时重复添加 (0, 1)
        if n > 2:
            edges.append((0, n - 1))
            
    elif edge_type == 'dense':
        # 稠密变化：所有可能的两两组合
        edges = list(combinations(range(n), 2))
        
    elif edge_type == 'firstlast':
        # 仅首尾变化：只看整体时间跨度的总变化
        edges = [(0, n - 1)]
        
    else:
        raise ValueError(f'Unknown edge type: {edge_type}')
        
    return edges


def get_ch(seg: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:
    ch = [torch.ne(seg[:, t1], seg[:, t2]) for t1, t2 in edges]
    ch = torch.stack(ch).transpose(0, 1)
    return ch


def load_json(file: Path) -> dict:
    with open(str(file)) as f:
        d = json.load(f)
    return d


def write_json(file: Path, data: dict) -> None:
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def active_learning_sampling(logits_seg: torch.Tensor, logits_ch: torch.Tensor, num_points: int = 100, nms_radius: int = 5):
    """
    根据模型的预测结果来采样需要标注的点，用于主动学习。
    包含 NMS 空间抑制与【Top-5% 高分池随机补齐】机制。
    """
    with torch.no_grad():
        p_seg = torch.sigmoid(logits_seg)
        p_ch = torch.sigmoid(logits_ch)    
        has_batch = True
        if p_seg.dim() == 4:
            has_batch = False
            p_seg = p_seg.unsqueeze(0)
            p_ch = p_ch.unsqueeze(0)
                
        B, T, C, H, W = p_seg.shape
        device = p_seg.device
            
        # 1. 置信度低得分
        unc_seg = 1.0 - torch.abs(2.0 * p_seg - 1.0)
        unc_ch = 1.0 - torch.abs(2.0 * p_ch - 1.0)
        score_unc = unc_seg.mean(dim=1) + unc_ch.mean(dim=1) 
            
        # 2. 预测不一致得分 (严格切片)
        p_seg_diff = torch.abs(p_seg[:, 1:] - p_seg[:, :-1]) 
        
        # 对齐变化头维度
        if p_ch.shape[1] == T:
            p_ch_aligned = p_ch[:, 1:]
        else:
            p_ch_aligned = p_ch
            
        inconsistency = torch.abs(p_seg_diff - p_ch_aligned)
        score_incons = inconsistency.mean(dim=1) 
            
        # 综合打分 (B, H, W)
        total_score = (score_unc + score_incons).mean(dim=1) 
            
        sampled_points = []
        for b in range(B):
            score_map = total_score[b].unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            # --- 步骤 A：利用 MaxPool2d 进行 NMS 空间抑制 ---
            kernel_size = nms_radius * 2 + 1
            local_max = F.max_pool2d(score_map, kernel_size=kernel_size, stride=1, padding=nms_radius)
            
            peak_mask = (score_map == local_max)
            peak_mask = peak_mask.squeeze() # (H, W)
            score_map_2d = score_map.squeeze() # (H, W)
            
            # 提取所有 NMS 候选点的坐标 [N, 2]
            candidate_coords = torch.nonzero(peak_mask)
            
            final_coords_list = []
            
            # --- 步骤 B：数量检查与【Top-5% 随机补齐】 ---
            if candidate_coords.shape[0] >= num_points:
                # NMS 候选点充足，直接从中挑取得分最高的 num_points 个
                candidate_scores = score_map_2d[candidate_coords[:, 0], candidate_coords[:, 1]]
                _, topk_indices = torch.topk(candidate_scores, num_points)
                final_coords_list = candidate_coords[topk_indices].tolist()
            else:
                # 状况 2：NMS 抑制太猛，点不够，启动 Top-5% 随机补齐
                if candidate_coords.shape[0] > 0:
                    final_coords_list = candidate_coords.tolist()
                
                need_count = num_points - len(final_coords_list)
                
                # 创建一个打分图副本，拉黑已经被选中的点
                temp_score_map = score_map_2d.clone()
                if len(final_coords_list) > 0:
                    temp_score_map[candidate_coords[:, 0], candidate_coords[:, 1]] = float('-inf')
                    
                flat_scores = temp_score_map.flatten()
                total_pixels = flat_scores.numel()
                
                # ---> 核心逻辑修改：构建高分候选池 <---
                
                # 计算 5% 的像素数量
                top_5_percent_count = int(total_pixels * 0.05)
                
                # 安全兜底：如果 5% 的数量比还需要补齐的 need_count 还要少
                # 强制将候选池大小扩大到 need_count，避免随机抽样越界
                pool_size = max(top_5_percent_count, need_count)
                pool_size = min(pool_size, total_pixels) # 防止超过总像素数
                
                # 1. 先找出全图得分排在前 pool_size 名的像素索引
                _, top_pool_indices = torch.topk(flat_scores, pool_size)
                
                # 2. 从这个高分池中，随机无放回地抽取 need_count 个
                rand_idx = torch.randperm(pool_size, device=device)[:need_count]
                chosen_flat_indices = top_pool_indices[rand_idx]
                
                # 3. 还原一维索引对应的 2D 坐标并加入最终列表
                for idx in chosen_flat_indices:
                    y = (idx // W).item()
                    x = (idx % W).item()
                    final_coords_list.append([y, x])
            
            # 格式化输出
            points = [(int(coord[0]), int(coord[1])) for coord in final_coords_list]
            
            if has_batch:
                sampled_points.append(points)
            else:
                sampled_points = points
                    
        return sampled_points, total_score

def generate_active_labels_from_points(sampled_points: list, gt_labels: torch.Tensor):
    """
    根据主动采样的坐标，从真实的密集标签中提取时序点监督信号（充当 Oracle）。
    
    Args:
        sampled_points: 采样的点集。如果是带有 batch 维度的，结构通常为 [[(y, x), ...], [(y, x), ...]]。
        gt_labels: 完整的真实密集标签，形状假定为 (B, T, C, H, W) 或 (B, T, H, W)。
        
    Returns:
        y_point_active: 形状与 gt_labels 一致的张量，仅在采样点处有真实值，其余位置为 0。
        loss_mask_active: 形状为 (B, 1, H, W) 的张量，仅在采样点处为 1.0，其余为 0。用于限制 Loss 的计算范围。
    """
    # 1. 维度兼容性对齐：确保 gt_labels 有通道维度 C
    if gt_labels.dim() == 4:
        gt_labels = gt_labels.unsqueeze(2) # (B, T, H, W) -> (B, T, 1, H, W)
        
    B, T, C, H, W = gt_labels.shape
    device = gt_labels.device
    
    # 2. 初始化全 0 的掩码和点标签
    loss_mask_active = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    y_point_active = torch.zeros_like(gt_labels, device=device)
    
    # 3. 兼容单图(Batch=1)传参时未嵌套列表的情况
    if B == 1 and len(sampled_points) > 0 and not isinstance(sampled_points[0], list):
        sampled_points = [sampled_points]
        
    # 4. 遍历提取 Oracle 真值
    for b in range(B):
        # 获取当前 batch 图的所有打点
        points = sampled_points[b]
        
        for (y, x) in points:
            # 安全检查：防止计算出的点坐标越界
            if 0 <= y < H and 0 <= x < W:
                # A. 激活该像素位置的 Loss 计算
                loss_mask_active[b, 0, y, x] = 1.0
                
                # B. 【核心动作：时序穿透提取】
                # 把该点在这 T 个月里所有的真实状态一次性全部提取出来
                y_point_active[b, :, :, y, x] = gt_labels[b, :, :, y, x]
                
    return y_point_active, loss_mask_active
