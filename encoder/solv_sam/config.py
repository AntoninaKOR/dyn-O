from dataclasses import dataclass, field
from typing import List, Optional, Literal

import torch


@dataclass
class SolvSamConfig:
    exp_name: str = "encoder_test"
    run_name: str = "test"

    # === Data Related Parameters ===
    dataset: str = "procgen_minari"
    root: str = field(default="/scratch/cluster/zzwang_new/procgen_data")

    # procgen minari dataset
    train_dataset_ids: List[str] = field(
        default_factory=lambda: ["procgen-starpilot-v4"],
    )
    valid_dataset_ids: List[str] = field(
        default_factory=lambda: ["procgen-starpilot-v1"],
    )

    # === ViT Related Parameters ===
    resize_to: List[int] = field(default_factory=lambda: [224, 224])
    encoder: Literal[
        "dinov2-vitb-14", "mae-vitb-16",
        "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-CI16x16",
    ] = "Cosmos-0.1-Tokenizer-CI16x16"

    # === Slot Attention Related Parameters ===
    num_slots: int = 47
    slot_att_iter: int = 3
    slot_dim: int = 256
    encode_use_mask: bool | None = None
    attn_loss_weight: float = 1.0

    # === Decode Related Parameters ===
    decode_segmentation: bool = False
    decode_use_mask: bool = True
    decoder_depth: int = 4

    # === SAM Scheduler Related Parameters ===
    # ratio of SAM masks that are not dropped, set to 1.0 to disable SAM mask dropout
    no_drop_ratio: float = 0.1

    schedule_type: Literal["log", "linear"] = "log"
    # fill the S slots x N patches mask, on the patch, slot, or frame level
    drop_type: Literal["patch", "slot", "frame"] = "patch"

    schedule_start_epoch: int = 2
    schedule_end_epoch: int = 14

    # drop_prob = max_drop_prob * log(1 + k * (epoch - start)) / log(1 + k * (end - start))
    log_schedule_k: float = 0.5

    # === Training Related Parameters ===
    learning_rate: float = 4e-4
    batch_size: int = 64
    num_epochs: int = 15

    # === Misc ===
    checkpoint_path: Optional[str] = None
    backup_per_second: int = 600
    validation_epoch: int = 1
    train_visualize_freq: int = 1000
    val_visualize_num_videos: int = 10
    val_visualize_num_frames_per_video: int = 10

    seed: int = 32

    # Derived attributes
    gpus: int = field(init=False)
    patch_size: int = field(init=False)
    token_num: int = field(init=False)

    def __post_init__(self):
        self.gpus = torch.cuda.device_count()
        if self.encoder.startswith("Cosmos"):
            self.patch_size = int(self.encoder.split("x")[-1])
            self.token_dim = 16
        else:
            self.patch_size = int(self.encoder.split("-")[2])
            self.token_dim = 768
        self.token_num = (self.resize_to[0] * self.resize_to[1]) // (self.patch_size ** 2)

        assert self.log_schedule_k > 0
        assert self.schedule_start_epoch < self.schedule_end_epoch
        assert 0 <= self.no_drop_ratio <= 1
