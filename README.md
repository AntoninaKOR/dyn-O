# Official Implementation of Dyn-O: Building Structured World Models with Object-Centric Representations (NeurIPS 2025)


## Installation
The code is validated on python 3.10.14 + CUDA 11.8 + pyTorch 2.4.0. It should work for newer python, CUDA and pyTorch versions but not guaranteed. 
```
conda create -n dynO python=3.10
git clone https://github.com/wangzizhao/dyn-O.git
cd dyn-O
pip install -r requirements.txt
```

## Procgen Data Collection
First, train the PPG policy.
```
python ppg_procgen.py \
    --env_id bigfish \
    --start_level 0 \
    --num_levels 250 \
    --distribution_mode hard
```
Then, collect training, validation, and test data with the learned policy.
```
CKPT_PATH="/path/to/checkpoint"
DATA_SAVE_PATH="/path/to/save/data"

python ppg_rollout_minari.py \
    --ckpt_path $CKPT_PATH \
    --data_root $DATA_SAVE_PATH \
    --data_version 0 \
    --start_level 0 \
    --num_levels 200 \
    --distribution_mode hard

python ppg_rollout_minari.py \
    --ckpt_path $CKPT_PATH \
    --data_root $DATA_SAVE_PATH \
    --data_version 1 \
    --start_level 200 \
    --num_levels 50 \
    --distribution_mode hard

python ppg_rollout_minari.py \
    --ckpt_path $CKPT_PATH \
    --data_root $DATA_SAVE_PATH \
    --data_version 2 \
    --start_level 250 \
    --num_levels 0 \
    --distribution_mode hard
```
Afterwards, preprocess training, validation, and test data with SAM. You can use multiple gpu for faster preprocessing by specifying `--cuda_ids`.
```
ENV_ID=bigfish
DATA_SAVE_PATH="/path/to/save/data"

for data_version in 0 1 2
do
    python segment-anything-2/video_track_all_obj.py \
        --data_path ${DATA_SAVE_PATH}/procgen-${ENV_ID}-v${data_version} \
        --cuda_ids 0
done
```

## Prepare Cosmos Tokenizer

Download [Cosmos Tokenizer Weight](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CI16x16) to 
```
├── encoder/cosmos/pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16/
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
```

## Training

If you encouter any error complaining missing episodes in `sam_masks.h5`, rerun the above preprocessing script. There is no need to remove existing `sam_masks.h5`, the script will resume from the existing file and only preprocess the missing episodes.
```
# train encoder
./train_encoder.sh
# train dynamics model
./train_dynamics.sh
```

