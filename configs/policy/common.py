import dataclasses
from typing import Literal


@dataclasses.dataclass
class MHAConfig:
    d_model: int = 512
    num_blocks: int = 2
    num_heads: int = 8


@dataclasses.dataclass
class WorldModelEnvConfig:
    deterministic: bool = False
    n_rollout_steps: int = 10
    random_static: bool | None = True
    random_one_slot: bool = True
    random_prob: float = 0.25
    obs_mode: Literal["enc_feat", "slots", "decode_rgb_then_enc_feat"] = None