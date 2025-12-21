#!/bin/bash

# conda activate torch-2.0-cr

ARCH_ALL="(32, 32, 32, 64, 64, 64, 64, 96, 96, 192, 192, 192, 192, 320, 320, 320, 320, 320, 320, 320, 768, 768, 768, 768, 768, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"

DATA_PATH=/mnt/data/ImageNet100/
N_GPUs=4
BS=224

export text_path=./1-output.txt

TYPE=stage1
VERSION=v1.2
job_dir="./outputs/$TYPE/$VERSION"
export DATA_PATH
export N_GPUs
export BS
export TYPE
export VERSION
export ARCH_ALL

model=reactnet_firstBConv_lbp_ther_48c_100_stage1
job_dir="./outputs/$TYPE/$VERSION"
mkdir -p $job_dir

EPOCHS=128

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=$N_GPUs \
--use_env --master_port=14484 stage1/train/main/distributed_amp.py \
--data_set imagenet_tensor \
--data_path $DATA_PATH \
--job_dir $job_dir/ \
--arch $model \
--lr 1.25e-3 \
--batch_size 32 \
--optimizer Adam \
--weight_decay 1e-5 \
--momentum 0.9 \
--epochs $EPOCHS \
--workers 4 \
--opt-level O0 \
2>&1 | tee $job_dir/log.txt
