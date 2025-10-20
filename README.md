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

## Training
```
# train encoder
./train_encoder.sh
# train dynamics model
./train_dynamics.sh
```

