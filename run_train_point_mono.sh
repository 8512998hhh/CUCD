#!/bin/bash
set -e
cd /root/autodl-tmp/ContUrbanCD-master

echo "=================================================="
echo "Point-Supervised + Temporal Smoothness Training"
echo "Date: $(date)"
echo "=================================================="

python train_point_mono.py \
    -c conturbancd_sn7_GLMamba_PointMono \
    -o GLMamba_PointMono_SN7_e50 \
    -d /root/autodl-tmp/SN7

echo "=================================================="
echo "Training Completed at: $(date)"
echo "=================================================="
