import dataclasses
from typing import Literal, List, Union

from configs.policy.common import MHAConfig, WorldModelEnvConfig


@dataclasses.dataclass
class ActorCriticModelConfig:
    input_mode: Literal["enc_feat", "slots"] = "enc_feat"
    downsample_with_conv: bool | None = True
    proj_in_dim: List[int] = dataclasses.field(default_factory=lambda: [1024])
    mha_cfg: MHAConfig = dataclasses.field(
        default_factory=lambda: MHAConfig(
            d_model=512,
            num_blocks=1,
            num_heads=8,
        ),
    )
    two_hot_rets: bool = True


@dataclasses.dataclass
class Training:
    use_imagination: bool | None = True
    imagination_prob: float = 1.0

    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    eps: float = 1e-5
    batch_size: int = 1024
    minibatch_size: int = 1024
    grad_norm_clip: float = 0.5
    target_update_tau: float = 0.02

    gamma: float = 0.995
    lambda_: float = 0.95
    entropy_weight: float = 0.01

    # only used when collecting imagination data
    action_uniform_prob: float = 0.05


@dataclasses.dataclass
class ActorCriticConfig:
    checkpoint_path: Union[str, None] = None

    actor_critic_model: ActorCriticModelConfig = dataclasses.field(default_factory=ActorCriticModelConfig)
    training: Training = dataclasses.field(default_factory=Training)
    world_model_env: WorldModelEnvConfig = dataclasses.field(
        default_factory=lambda: WorldModelEnvConfig(
            deterministic=True,
            n_rollout_steps=15,
            random_static=True,
            random_one_slot=True,
            random_prob=0.25,
        ),
    )

    update_frequency: int = 1

    def __post_init__(self):
        self.world_model_env.obs_mode = self.actor_critic_model.input_mode
