#!/bin/bash

# experiment tracking: auto (wandb if installed, else mlflow), wandb, comet or mlflow.
# comet reads COMET_API_KEY and COMET_WORKSPACE from the environment.
LOGGER="comet"

# visualisations always land in results/${EXP_NAME}/${RUN_NAME}_<timestamp>/viz. Uploads
# are shrunk to a jpeg first, but set this to False where the asset endpoint is slow
# enough that even that stalls.
UPLOAD_IMAGES=True

# EXP_NAME is the project, RUN_NAME the run within it; both loggers get a timestamp
# appended to RUN_NAME so repeated runs do not collide
EXP_NAME="dynO"
RUN_NAME="encoder_homegrid"

# to continue an interrupted run, point this at its checkpoint_latest.pt or
# checkpoint_epoch_N.pt. Weights, optimiser and schedule are restored and the epoch is
# derived from the iteration count. A bare relaunch cannot find it on its own, since the
# run directory carries the launch timestamp.
RESUME_FROM=""

# empty starts a fresh experiment
COMET_EXPERIMENT_KEY=""

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

FEAT_LOSS_WEIGHT=1.0
RGB_LOSS_WEIGHT=1.0

# encoder
# homegrid frames hold 5 entities on average and at most 10, floor and walls included;
# without masks the decoder reconstructs those too, so they occupy slots as well
NUM_SLOTS=10
# paper setting: 224x224 over a patch of 16 gives 196 tokens. resize_to must be divisible
# by the patch size, which config.py parses out of the encoder name
ENCODER="Cosmos-0.1-Tokenizer-CI16x16"
RESIZE_TO="224 224"
BATCH_SIZE=64                               # global; the dataloader splits it across GPUS

# physical GPUs to run on, comma separated: "0", "3", "2,5", ...
GPU_IDS="0"

NUM_EPOCHS=50

# window over which SAM masks are faded out, kept at 10% to 90% of the run. Inert while
# LOAD_SAM_MASKS is False, since that pins no_drop_ratio to 1.0 and skips the schedule
SCHEDULE_START_EPOCH=5
SCHEDULE_END_EPOCH=45

# tyro renders plain bools as a pair of flags rather than as a flag taking a value
if [ "${LOAD_SAM_MASKS}" = "True" ]; then
  LOAD_SAM_MASKS_FLAG="--load_sam_masks"
else
  LOAD_SAM_MASKS_FLAG="--no_load_sam_masks"
fi

if [ "${UPLOAD_IMAGES}" = "True" ]; then
  UPLOAD_IMAGES_FLAG="--upload_images"
else
  UPLOAD_IMAGES_FLAG="--no_upload_images"
fi

if [ -n "${RESUME_FROM}" ]; then
  RESUME_FLAG="--checkpoint_path ${RESUME_FROM}"
else
  RESUME_FLAG=""
fi

GPUS=$(awk -F, '{print NF}' <<< "${GPU_IDS}")

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --master_port=${MASTER_PORT:-12345} --nproc_per_node=${GPUS} encoder/solv_sam/train.py \
  --exp_name ${EXP_NAME} \
  --run_name ${RUN_NAME} \
  --logger ${LOGGER} \
  --comet_experiment_key "${COMET_EXPERIMENT_KEY}" \
  --num_slots ${NUM_SLOTS} \
  --encoder ${ENCODER} \
  --resize_to ${RESIZE_TO} \
  --batch_size ${BATCH_SIZE} \
  --root ${DATA_ROOT} \
  --train_dataset_ids ${DATASET_ID}-${ENVID}-v${TRAIN_VERSION} \
  --valid_dataset_ids ${DATASET_ID}-${ENVID}-v${VALID_VERSION} \
  ${LOAD_SAM_MASKS_FLAG} \
  ${UPLOAD_IMAGES_FLAG} \
  --encode_use_mask ${ENCODE_USE_MASK} \
  --no_drop_ratio ${NO_DROP_RATIO} \
  --attn_loss_weight ${ATTN_LOSS_WEIGHT} \
  --feat_loss_weight ${FEAT_LOSS_WEIGHT} \
  --rgb_loss_weight ${RGB_LOSS_WEIGHT} \
  --num_epochs ${NUM_EPOCHS} \
  --schedule_start_epoch ${SCHEDULE_START_EPOCH} \
  --schedule_end_epoch ${SCHEDULE_END_EPOCH} \
  ${RESUME_FLAG}
