from typing import Union, Any
from dataclasses import dataclass, field

from configs.data import DataConfig
from configs.dynamics import DynamicsConfig
from configs.encoder import EncoderConfig
from configs.env import EnvConfig
from configs.policy import PolicyConfig
from configs.logging import LoggingConfig


@dataclass
class Config:
    # ================= Misc =================
    exp_name: str = "oc_ssm"
    run_name: str = "default"
    seed: int = 42
    distributed: bool | None = True
    cuda_id: int = 0
    dtype_name: str = "float32"
    torch_deterministic: bool = True
    debug: bool = True

    # ================= Model =================
    env_type: str = "procgen"
    encoder_type: str = "solv_sam"
    dynamics_type: str = "mamba"
    policy_type: str = "actor_critic"
    replay_buffer_type: str = "offline"

    # ================= Training =================
    total_timesteps: int = int(1e6)
    learning_starts: int = int(1e4)     # disabled for offline
    update_dynamics: bool = True
    update_policy: bool = False

    # if update_policy is False, will still evaluate policy when eval_policy is True
    eval_policy: bool = False
    policy_learning_starts: int = 0

    # ============== Env Wrapper ==============
    # TODO: if true, need to set up ffmpeg
    capture_video: bool = False
    reward_normalization_gamma: float = 0.999

    # ============== Dynamically Set Config ==============
    offline: bool = field(init=False)

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data_all: DataConfig = field(default_factory=DataConfig)
    dynamics_all: DynamicsConfig = field(default_factory=DynamicsConfig)
    encoder_all: EncoderConfig = field(default_factory=EncoderConfig)
    env_all: EnvConfig = field(default_factory=EnvConfig)
    policy_all: PolicyConfig = field(default_factory=PolicyConfig)

    data: Any = field(init=False)
    dynamics: Any = field(init=False)
    encoder: Any = field(init=False)
    env: Any = field(init=False)
    policy: Any = field(init=False)

    def __post_init__(self):
        self.offline = self.replay_buffer_type == "offline"

        # assign configs based on the type
        self.data = self.data_all.configs[self.replay_buffer_type]
        self.dynamics = self.dynamics_all.configs[self.dynamics_type]
        self.encoder = self.encoder_all.configs[self.encoder_type]
        self.env = self.env_all.configs[self.env_type]
        self.policy = self.policy_all.configs[self.policy_type]

        if self.dynamics.patch_as_slot:
            self.dynamics.num_slots = self.encoder.token_num

        if self.update_dynamics and not self.encoder.use_sam_mask and not self.dynamics.patch_as_slot:
            assert self.dynamics.num_slots == self.encoder.num_slots
