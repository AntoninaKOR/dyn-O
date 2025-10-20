import dataclasses
from typing import Literal, List, Union

from configs.policy.common import MHAConfig, WorldModelEnvConfig


@dataclasses.dataclass
class ModelConfig:
    input_mode: Literal["enc_feat", "slots"] = "enc_feat"
    proj_in_dim: List[int] = dataclasses.field(default_factory=lambda: [512])
    mha_cfg: MHAConfig = dataclasses.field(
        default_factory=lambda: MHAConfig(
            d_model=512,
            num_blocks=2,
            num_heads=8,
        ),
    )


@dataclasses.dataclass
class Training:
    lr: float = 3e-4
    batch_size: int = 64
    grad_norm_clip: float = 0.5


@dataclasses.dataclass
class BehaviorCloningConfig:
    checkpoint_path: Union[str, None] = None

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: Training = dataclasses.field(default_factory=Training)
    world_model_env: WorldModelEnvConfig = dataclasses.field(default_factory=WorldModelEnvConfig)

    update_frequency: int = 1

    def __post_init__(self):
        self.world_model_env.obs_mode = self.model.input_mode
