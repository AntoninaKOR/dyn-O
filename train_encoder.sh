#!/bin/bash

# experiment tracking: auto (wandb if installed, else mlflow), wandb, comet or mlflow.
# comet reads COMET_API_KEY and COMET_WORKSPACE from the environment.
LOGGER="comet"

# EXP_NAME is the project, RUN_NAME the run within it; both loggers get a timestamp
# appended to RUN_NAME so repeated runs do not collide
EXP_NAME="dynO"
RUN_NAME="encoder_homegrid"

# data
DATA_ROOT="/data/minari"
DATASET_ID="dyno"
ENVID="homegrid"
TRAIN_VERSION=0
VALID_VERSION=1

# set to False to train fully unsupervised: sam_masks.h5 is never read, and the mask
# schedule, the attention loss and the segmentation metrics are switched off
LOAD_SAM_MASKS=False

ENCODE_USE_MASK=False
NO_DROP_RATIO=0.1
ATTN_LOSS_WEIGHT=0.0

# encoder
# homegrid frames hold 5 entities on average and at most 10, floor and walls included;
# without masks the decoder reconstructs those too, so they occupy slots as well
NUM_SLOTS=10
# paper setting: 224x224 over a patch of 16 gives 196 tokens. resize_to must be divisible
# by the patch size, which config.py parses out of the encoder name
ENCODER="Cosmos-0.1-Tokenizer-CI16x16"
RESIZE_TO="224 224"
BATCH_SIZE=64                               # global; the dataloader splits it across GPUS

GPUS=1

NUM_EPOCHS=10
SCHEDULE_START_EPOCH=1
SCHEDULE_END_EPOCH=9

# tyro renders plain bools as a pair of flags rather than as a flag taking a value
if [ "${LOAD_SAM_MASKS}" = "True" ]; then
  LOAD_SAM_MASKS_FLAG="--load_sam_masks"
else
  LOAD_SAM_MASKS_FLAG="--no_load_sam_masks"
fi

# built by hand rather than with seq, whose separator handling differs between platforms
GPU_IDS=0
for ((i = 1; i < GPUS; i++)); do GPU_IDS="${GPU_IDS},${i}"; done

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --master_port=12345 --nproc_per_node=${GPUS} encoder/solv_sam/train.py \
  --exp_name ${EXP_NAME} \
  --run_name ${RUN_NAME} \
  --logger ${LOGGER} \
  --num_slots ${NUM_SLOTS} \
  --encoder ${ENCODER} \
  --resize_to ${RESIZE_TO} \
  --batch_size ${BATCH_SIZE} \
  --root ${DATA_ROOT} \
  --train_dataset_ids ${DATASET_ID}-${ENVID}-v${TRAIN_VERSION} \
  --valid_dataset_ids ${DATASET_ID}-${ENVID}-v${VALID_VERSION} \
  ${LOAD_SAM_MASKS_FLAG} \
  --encode_use_mask ${ENCODE_USE_MASK} \
  --no_drop_ratio ${NO_DROP_RATIO} \
  --attn_loss_weight ${ATTN_LOSS_WEIGHT} \
  --num_epochs ${NUM_EPOCHS} \
  --schedule_start_epoch ${SCHEDULE_START_EPOCH} \
  --schedule_end_epoch ${SCHEDULE_END_EPOCH}
