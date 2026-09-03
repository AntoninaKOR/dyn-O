from dataclasses import dataclass, field
from typing import List, Optional, Literal

import torch


@dataclass
class SolvSamConfig:
    exp_name: str = "encoder_test"
    run_name: str = "test"

    # Experiment tracking backend. "auto" keeps the historical behaviour of using wandb
    # when it is importable and falling back to mlflow otherwise.
    logger: Literal["auto", "wandb", "comet", "mlflow"] = "auto"

    # Comet sends assets to a different endpoint than metrics, and clusters that firewall
    # it make every visualisation stall for the full upload timeout. Visualisations are
    # written under ckpt_path either way, so uploading them is optional.
    upload_images: bool = True

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

    # === SAM Mask Related Parameters ===
    # When False, sam_masks.h5 is never read and the encoder trains fully unsupervised:
    # slot attention gets no mask, the attention loss is disabled and the RGB loss covers
    # every pixel. Segmentation metrics need ground truth, so they are unavailable.
    # Deliberately not called use_sam_mask: that name exists in the inference-side config
    # and the two are compared by common key when a checkpoint is loaded.
    load_sam_masks: bool = True

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

    feat_loss_weight: float = 1.0
    rgb_loss_weight: float = 1.0

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

        assert self.feat_loss_weight >= 0 and self.rgb_loss_weight >= 0
        if self.feat_loss_weight == 0 and self.rgb_loss_weight == 0:
            raise ValueError(
                "feat_loss_weight and rgb_loss_weight are both zero, nothing would train"
            )

        if not self.load_sam_masks:
            if self.decode_segmentation:
                raise ValueError(
                    "decode_segmentation needs ground-truth masks as its target; "
                    "it cannot be combined with load_sam_masks=False."
                )

            # every mask-dependent knob defaults to a value that assumes masks exist, so
            # override them rather than forcing the caller to repeat the whole combination
            overridden = {}
            if self.encode_use_mask:
                overridden["encode_use_mask"] = False
            if self.decode_use_mask:
                overridden["decode_use_mask"] = False
            if self.attn_loss_weight != 0.0:
                overridden["attn_loss_weight"] = 0.0
            if self.no_drop_ratio != 1.0:
                overridden["no_drop_ratio"] = 1.0

            for name, value in overridden.items():
                setattr(self, name, value)
            if overridden:
                print(f"load_sam_masks=False, overriding {overridden}")
