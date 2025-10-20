from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class SolvSamConfig:
    # === Misc ===
    checkpoint_path: Optional[str] = None

    # === ViT Related Parameters ===
    resize_to: List[int] = field(default_factory=lambda: [224, 224])
    encoder: Literal[
        "dinov2-vitb-14", "mae-vitb-16",
        "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-CI16x16",
    ] = "Cosmos-0.1-Tokenizer-CI16x16"

    use_sam_mask: bool = False

    # === Slot Attention Related Parameters ===
    num_slots: int = 31
    slot_att_iter: int = 3
    slot_dim: int = 256

    # === Decode Related Parameters ===
    decode_segmentation: bool = False
    decoder_depth: int = 4

    # === Training Related Parameters ===
    finetune: bool = False
    learning_rate: float = 4e-4
    batch_size: int = 64

    # Derived attributes
    patch_size: int = field(init=False)
    token_num: int = field(init=False)

    def __post_init__(self):
        if self.encoder.startswith("Cosmos"):
            self.patch_size = int(self.encoder.split("x")[-1])
            self.token_dim = 16
        else:
            self.patch_size = int(self.encoder.split("-")[2])
            self.token_dim = 768
        self.token_num = (self.resize_to[0] * self.resize_to[1]) // (self.patch_size ** 2)
