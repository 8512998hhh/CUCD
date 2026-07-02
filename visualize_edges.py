import cv2 as cv
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import sys
import os

def detect_edge(image):
    """使用Sobel算子提取单通道图像的边缘"""
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    edge_intensity = np.sqrt(dy**2 + dx**2)
    return edge_intensity

def detect_dynamic_edge_v2(image_np):
    """
    针对小尺度建筑优化的边缘提取
    注意：OpenCV 的高级滤波和 Canny 通常需要 uint8 格式输入
    """
    # 1. 格式转换: 假设传入的 image_np 是 0~1 的 float32
    img_8u = np.uint8(np.clip(image_np * 255, 0, 255))
    
    # 2. 🔥 双边滤波 (Bilateral Filter)：保边降噪的王者
    # d=5: 邻域直径; sigmaColor/sigmaSpace: 越大越模糊，但遇到强边缘会保留
    blurred = cv.bilateralFilter(img_8u, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 3. 🔥 Canny 边缘检测：连接断裂边缘，去除杂乱弱梯度
    # 50 和 150 是双阈值，低于 50 的绝对被抛弃，高于 150 的绝对是边缘
    # 在 50~150 之间的，只有与强边缘相连才被保留 (这非常适合连接房子的直角)
    edges = cv.Canny(blurred, 50, 150)
    
    # 4. 转换回 0-1 的浮点数掩膜
    edges_float = (edges > 0).astype(np.float32)
    
    # (可选) 如果你觉得 Canny 提取的线太细(单像素)，网络不好学，可以稍微膨胀一下
    # kernel = np.ones((2, 2), np.uint8)
    # edges_float = cv.dilate(edges_float, kernel, iterations=1)
    
    return edges_float

def extract_building_edges_multispectral(img_t, red_idx=2, nir_idx=3, ndvi_thresh=0.2):
    """
    集成了双边滤波、Canny 算子、多波段融合以及 NDVI 掩膜的终极边缘提取函数。
    
    Args:
        img_t: 单个时间步的多光谱图像，形状为 (C, H, W)，浮点数 [0, 1]
        red_idx: 红波段的通道索引 (默认 2)
        nir_idx: 近红外波段的通道索引 (默认 3)
        ndvi_thresh: NDVI 阈值，高于此值被判定为植被并抹除边缘
        
    Returns:
        np.ndarray: 形状为 (H, W) 的纯净二值边缘图
    """
    C, H, W = img_t.shape
    
    # ==========================================
    # 1. 提取全通道的结构边缘 (Canny)
    # ==========================================
    edge_channels = []
    for c in range(C):
        channel_img = img_t[c]
        
        # 格式转换
        img_8u = np.uint8(np.clip(channel_img * 255, 0, 255))
        
        # 双边滤波 (保边降噪)
        blurred = cv.bilateralFilter(img_8u, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Canny 算子
        edges = cv.Canny(blurred, 50, 150)
        
        # 转为 0 和 1 的浮点数
        edge_channels.append((edges > 0).astype(np.float32))
        
    # 跨波段逻辑或融合：只要任何一个波段有边缘，就保留
    edge_channels = np.stack(edge_channels, axis=0)
    combined_edges = np.max(edge_channels, axis=0) # 形状: (H, W)
    
    # ==========================================
    # 2. 计算 NDVI 掩膜 (降维打击植被)
    # ==========================================
    red_band = img_t[red_idx]
    nir_band = img_t[nir_idx]
    
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    
    # 将 NDVI 大于阈值的地方（农田/草地/树林）的边缘强行抹零
    combined_edges[ndvi > ndvi_thresh] = 0.0
    
    # ==========================================
    # 3. (可选) 边缘膨胀：让网络更好学
    # ==========================================
    # Canny 提取的线是绝对的单像素(1-pixel width)，在深度学习中 1 pixel 的线极易产生梯度消失。
    # 强烈建议用一个 2x2 的核稍微膨胀一下，变成 2 pixel 宽的线。
    kernel = np.ones((2, 2), np.uint8)
    combined_edges = cv.dilate(combined_edges, kernel, iterations=1)
    
    return combined_edges

def image_linear_transform(image, scale=[2, 98]):
    """基于百分位数的线性拉伸归一化"""
    image = np.float32(image)
    for i in range(image.shape[0]):
        min_, max_ = np.percentile(image[i], scale)
        image[i] = (image[i] - min_) / (max_ - min_ + 1e-7)
    return np.clip(image, 0, 1)

def extract_and_visualize(image_path):
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 '{image_path}'")
        return

    print(f"正在加载图像: {image_path} ...")
    img = tifffile.imread(image_path)
    
    # 归一化到 [0, 1] 范围
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.max() > 1.0:
        img = img.astype(np.float32) / 255.0 
        
    # 确保形状为 (C, H, W) 以便按通道处理
    # 通常 tifffile 读取的多光谱图像形状为 (H, W, C)
    if img.ndim == 3 and img.shape[2] <= 4: 
        img_chw = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img_chw = img[np.newaxis, :, :]
    else:
        img_chw = img
        
    C, H, W = img_chw.shape
    print(f"处理的图像形状 (C, H, W): {img_chw.shape}")

    # --- 提取轮廓 ---
    print("正在提取轮廓...")
    # 1. 在每个通道上应用 Sobel 算子
    # edge_intensity = np.stack([extract_building_edges_multispectral(img_chw[c]) for c in range(C)], axis=0)
    edge_intensity = extract_building_edges_multispectral(img_chw)
    # 2. 线性拉伸归一化
    edge_intensity = image_linear_transform(edge_intensity, [2, 98])
    # 3. 求通道平均
    # edge_intensity = np.mean(edge_intensity, axis=0)
    edge_intensity = np.max(edge_intensity, axis=0)
    # 4. 处理异常值
    edge_intensity[np.isnan(edge_intensity)] = 0
    edge_intensity[(edge_intensity < 0) | (edge_intensity > 1)] = 0
    # 5. 二值化 (0表示非轮廓，1表示轮廓)
    binary_edge = (edge_intensity > 0.5).astype(np.float32)

    # --- 可视化 ---
    print("正在生成可视化结果...")
    plt.figure(figsize=(12, 6))
    
    # 子图1: 原始图像 (RGB或灰度)
    plt.subplot(1, 2, 1)
    if C >= 3:
        # 提取前3个通道作为RGB进行显示，并做简单的拉伸以增强对比度
        rgb_img = np.transpose(img_chw[:3, :, :], (1, 2, 0))
        p98 = np.percentile(rgb_img, 98)
        rgb_img = np.clip(rgb_img / (p98 + 1e-7), 0, 1)
        plt.imshow(rgb_img)
    else:
        plt.imshow(img_chw[0], cmap='gray')
    plt.title("Original Image (RGB)")
    plt.axis('off')
    
    # 子图2: 提取的二值轮廓
    plt.subplot(1, 2, 2)
    plt.imshow(binary_edge, cmap='gray')
    plt.title("Extracted Binary Edges")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    save_path = "./edge_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
        extract_and_visualize(target_image)
    else:
        print("用法: python visualize_edges.py <tif文件路径>")
        print("示例: python visualize_edges.py sample.tif")
