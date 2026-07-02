#!/bin/bash

# 确保脚本在发生错误时退出
set -e

# 激活你的虚拟环境 (如果需要的话，请取消注释并修改路径)
# conda activate conturbancd

# 设置工作目录为项目根目录
cd /root/autodl-tmp/ContUrbanCD-master

echo "=================================================="
echo "Point-Supervised + Dense-Consistency Joint Training"
echo "Date: $(date)"
echo "=================================================="

# 点监督 + dense 一致性联合训练
# -c: 配置文件 (configs/conturbancd_sn7_GLMamba_PointAux.yaml)
# -o: 输出目录 (权重 / 评估结果)
# -d: 数据集根目录
python train_point_ssl_aux.py \
    -c conturbancd_sn7_GLMamba_PointAux \
    -o GLMamba_PointAux_SN7_e50 \
    -d /root/autodl-tmp/SN7

echo "=================================================="
echo "Training Completed at: $(date)"
echo "=================================================="
