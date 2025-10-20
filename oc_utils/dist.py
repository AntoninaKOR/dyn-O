import os
import sys

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(config):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        config.rank = int(os.environ['SLURM_PROCID'])
        config.world_size = config.gpus
        config.gpu = config.rank % torch.cuda.device_count()

    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        config.rank, config.gpu, config.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=config.world_size,
        rank=config.rank,
    )

    torch.cuda.set_device(config.gpu)
    print('| distributed init (rank {}): {}'.format(config.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(config.rank == 0)
