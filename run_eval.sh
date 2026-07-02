#!/bin/bash

# 确保脚本在发生错误时退出
set -e

# 激活你的虚拟环境 (如果需要的话，请取消注释并修改路径)
# source /path/to/your/venv/bin/activate
# 或者如果是 conda 环境:
# conda activate your_env_name

# 设置工作目录为项目根目录
cd /root/autodl-tmp/ContUrbanCD-master

# 打印开始信息
echo "=================================================="
echo "Starting Point-Supervised Training for ContUrbanCD"
echo "Date: $(date)"
echo "=================================================="

# 运行点监督训练脚本
# -c: 配置文件名称 (对应 configs/conturbancd_sn7_GLMamba.yaml)
# -o: 输出目录
# -d: 数据集路径
python eval.py \
    -c conturbancd_sn7_GLMamba_Point \
    -o GLMamba_Point_CL_SN7_e50_20260629\
    -d /root/autodl-tmp/SN7

# 打印结束信息
echo "=================================================="
echo "Training Completed at: $(date)"
echo "=================================================="
