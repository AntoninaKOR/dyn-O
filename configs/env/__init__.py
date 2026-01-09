from dataclasses import dataclass, field

from configs.env.procgen import ProcgenConfig


@dataclass
class EnvConfig:
    procgen: ProcgenConfig = field(default_factory=ProcgenConfig)

    def __post_init__(self):
        self.configs = {
            "procgen": self.procgen,
        }
