# Self-supervised Object-Centric Learning for Videos

This is modified from *Self-supervised Object-Centric Learning for Videos* ([Paper](https://arxiv.org/abs/2310.06907) | [Webpage](https://kuis-ai.github.io/solv)).

## Installation
Follow the instructions in `oc_ssm/README.md` to install the required dependencies.


## Training
```
torchrun --master_port=12345 --nproc_per_node=#gpus train.py \
--root /path/to/root \
--train_dataset_ids xxx xxx xxx \
--valid_dataset_ids xxx xxx xxx
```
For example:
```
torchrun --master_port=12345 --nproc_per_node=4 train.py \
--root /scratch/cluster/zzwang_new/procgen_data \
--train_dataset_ids procgen-bigfish-v0 procgen-bossfight-v0 procgen-plunder-v0 \
--valid_dataset_ids procgen-bigfish-v1 procgen-bossfight-v1 procgen-plunder-v1
```
