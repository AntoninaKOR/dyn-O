from dataclasses import dataclass, field

from configs.dynamics.mamba import MambaConfig


@dataclass
class DynamicsConfig:
    mamba: MambaConfig = field(default_factory=MambaConfig)

    def __post_init__(self):
        self.configs = {
            "mamba": self.mamba,
        }
