#!/bin/bash

EXP_NAME="dynO"

# data
DATA_ROOT="/scratch/cluster/zzwang_new/procgen_data"
DATASET_ID="procgen"
ENVID="bigfish"                             # bigfish, caveflyer, coinrun, dodgeball, jumper, ninja, starpilot, clevr, boxing, skiing

ENCODE_USE_MASK=True
NO_DROP_RATIO=0.1
ATTN_LOSS_WEIGHT=0.0

# encoder
NUM_SLOTS=31

NUM_EPOCHS=10
SCHEDULE_START_EPOCH=1
SCHEDULE_END_EPOCH=9

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=12345 --nproc_per_node=4 encoder/solv_sam/train.py \
  --exp_name ${EXP_NAME} \
  --run_name encoder_${ENVID} \
  --num_slots ${NUM_SLOTS} \
  --root ${DATA_ROOT} \
  --train_dataset_ids ${DATASET_ID}-${ENVID}-v0 \
  --valid_dataset_ids ${DATASET_ID}-${ENVID}-v4 \
  --encode_use_mask ${ENCODE_USE_MASK} \
  --no_drop_ratio ${NO_DROP_RATIO} \
  --attn_loss_weight ${ATTN_LOSS_WEIGHT} \
  --num_epochs ${NUM_EPOCHS} \
  --schedule_start_epoch ${SCHEDULE_START_EPOCH} \
  --schedule_end_epoch ${SCHEDULE_END_EPOCH} 