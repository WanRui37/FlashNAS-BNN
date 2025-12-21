#!/bin/bash

# 默认参数设置
DATASET="imagenet100"
GPU="0"
MODEL_ARCH="v_SVR"
MODE="test"
EPOCHES=256
BATCH_SIZE=1
LR=1e-4
EMBED_DIM=20
HIDDEN_SIZE=256
DROPOUT=0
RATIO=0.8
OUT_PATH="/mnt/wr/FlashNAS-BNN/net_predict_output/"
SEED=0


python ./net_predict/3-train_acc_predict.py \
    --model_arch "$MODEL_ARCH" \
    --dataset "$DATASET" \
    --gpu "$GPU" \
    --mode "$MODE" \
    --epoches "$EPOCHES" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --embed_dim "$EMBED_DIM" \
    --hidden_size "$HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --ratio "$RATIO" \
    --out_path "$OUT_PATH" \
    --seed "$SEED"
