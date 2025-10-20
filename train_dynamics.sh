#!/bin/bash

# misc
EXP_NAME="dynO"
SEED=0

# data
DATA_ROOT="/scratch/cluster/zzwang_new/procgen_data"
ENVID="jumper"                # bigfish, caveflyer, coinrun, dodgeball, jumper, ninja, starpilot, clevr, boxing, skiing
DATA_VERSION=0                  # 0: training, 1: validation, 2: test
TRAIN_RATIO=0.9

# encoder
ENCODER_PATH="/path/to/encoder/ckpt"
NUM_SLOTS=31                    # bigfish: 31, caveflyer: 63, coinrun: 31, dodgeball: 31, jumper: 31, ninja:47, starpilot: 47, clevr: 4, boxing: 15, skiing: 15
SLOT_DIM=256

# dynamics
BACKBONE="Mamba2"                # MHA, Mamba1, Mamba2

D_MODEL=512
N_LAYER=2
D_STATE=64
NUM_SLOT_MODES=1

# prediction
UPDATE_PRED=True
USE_PRETRAINED_DYNAMICS=False

N_PRED_STEPS=5
PRED_DELTA=True
LOSS_ONLY_USE_LAST_PRED=False
USE_REWARD_TERMINATION_WEIGHT=False
TERMINATION_PROB=-1

PATCH_AS_SLOT=False

PRED_ENC_FEAT=True
PRED_OBS=True
PRED_RATIO=0.1

# static-dynamic disentanglement
DISENTANGLE_STATIC_DYNAMIC=False
NORMALIZE_DYNAMIC_EMBEDDINGS=False
STATIC_DYNAMIC_MERGE_METHOD="concat"

# static feature
DISENTANGLE_SEQ_LEN=5
STATIC_CONTRASTIVE_COEF=1.0
STATIC_INVARIANCE_COEF=0.0

# dynamic feature
PROJ_IN_WEIGHT_DECAY=0
DYNAMIC_RECON_DROPOUT=0

DYNAMIC_CONTRASTIVE_COEF=0.0
DYNAMIC_ADVERSARIAL_LOSS_WEIGHT_SCHEDULE_END_VALUE=0.0
DYNAMIC_DISENTANGLEMENT_LR=1e-4

DYNAMIC_USE_VQ=False
CODEBOOK_SIZE=16384

ITERATIVE_DYNAMIC_TRAINING=False
USE_CA_GRAD=False
CA_GRAD_ALPHA=0.05
CA_GRAD_RESCALE=False

# training
TOTAL_TIMESTEPS=200000
BATCH_SIZE=32

# python run.py --distributed False \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=12345 --nproc_per_node=4 run.py \
  --seed ${SEED} \
  --exp_name ${EXP_NAME} \
  --run_name ${ENVID}_dynamics \
  --update_dynamics \
  --no-update_policy \
  --total_timesteps ${TOTAL_TIMESTEPS} \
  --data_all.offline.data_path ${DATA_ROOT} \
  --data_all.offline.dataset_ids procgen-${ENVID}-v${DATA_VERSION} \
  --data_all.offline.encoding_fname ${ENCODER_TYPE}/encoding.h5 \
  --data_all.offline.encoding_for_kmeans_fname ${ENCODER_TYPE}/encoding_for_kmeans.h5 \
  --data_all.offline.train_ratio ${TRAIN_RATIO} \
  --encoder_all.solv_sam.num_slots ${NUM_SLOTS} \
  --encoder_all.solv_sam.slot_dim ${SLOT_DIM} \
  --encoder_all.solv_sam.checkpoint_path ${ENCODER_PATH} \
  --dynamics_all.mamba.mamba.layer ${BACKBONE} \
  --dynamics_all.mamba.patch_as_slot ${PATCH_AS_SLOT} \
  --dynamics_all.mamba.num_slots ${NUM_SLOTS} \
  --dynamics_all.mamba.mamba.d_model ${D_MODEL} \
  --dynamics_all.mamba.mamba.n_layer ${N_LAYER} \
  --dynamics_all.mamba.mamba.ssm_cfg.d_state ${D_STATE} \
  --dynamics_all.mamba.pred.pred_delta ${PRED_DELTA} \
  --dynamics_all.mamba.pred.disentangle_static_dynamic ${DISENTANGLE_STATIC_DYNAMIC} \
  --dynamics_all.mamba.pred.normalize_dynamic_embeddings ${NORMALIZE_DYNAMIC_EMBEDDINGS} \
  --dynamics_all.mamba.pred.static_dynamic_merge_method ${STATIC_DYNAMIC_MERGE_METHOD} \
  --dynamics_all.mamba.pred.dynamic_use_vq ${DYNAMIC_USE_VQ} \
  --dynamics_all.mamba.pred.codebook_size ${CODEBOOK_SIZE} \
  --dynamics_all.mamba.pred.num_slot_modes ${NUM_SLOT_MODES} \
  --dynamics_all.mamba.training.batch_size ${BATCH_SIZE} \
  --dynamics_all.mamba.training.termination_prob ${TERMINATION_PROB} \
  --dynamics_all.mamba.training.update_pred ${UPDATE_PRED} \
  --dynamics_all.mamba.training.n_pred_steps ${N_PRED_STEPS} \
  --dynamics_all.mamba.training.use_pretrained_dynamics ${USE_PRETRAINED_DYNAMICS} \
  --dynamics_all.mamba.training.loss_only_use_last_pred ${LOSS_ONLY_USE_LAST_PRED} \
  --dynamics_all.mamba.training.use_reward_termination_weight ${USE_REWARD_TERMINATION_WEIGHT} \
  --dynamics_all.mamba.training.pred_enc_feat ${PRED_ENC_FEAT} \
  --dynamics_all.mamba.training.pred_obs ${PRED_OBS} \
  --dynamics_all.mamba.training.pred_ratio ${PRED_RATIO} \
  --dynamics_all.mamba.training.disentangle_seq_len ${DISENTANGLE_SEQ_LEN} \
  --dynamics_all.mamba.training.static_contrastive_coef ${STATIC_CONTRASTIVE_COEF} \
  --dynamics_all.mamba.training.static_invariance_coef ${STATIC_INVARIANCE_COEF} \
  --dynamics_all.mamba.training.proj_in_weight_decay ${PROJ_IN_WEIGHT_DECAY} \
  --dynamics_all.mamba.training.dynamic_recon_dropout ${DYNAMIC_RECON_DROPOUT} \
  --dynamics_all.mamba.training.dynamic_contrastive_coef ${DYNAMIC_CONTRASTIVE_COEF} \
  --dynamics_all.mamba.training.dynamic_adversarial_loss_weight_schedule.end_value ${DYNAMIC_ADVERSARIAL_LOSS_WEIGHT_SCHEDULE_END_VALUE} \
  --dynamics_all.mamba.training.dynamic_disentanglement_lr ${DYNAMIC_DISENTANGLEMENT_LR} \
  --dynamics_all.mamba.training.iterative_dynamic_training ${ITERATIVE_DYNAMIC_TRAINING} \
  --dynamics_all.mamba.training.use_ca_grad ${USE_CA_GRAD} \
  --dynamics_all.mamba.training.ca_grad_alpha ${CA_GRAD_ALPHA} \
  --dynamics_all.mamba.training.ca_grad_rescale ${CA_GRAD_RESCALE} \