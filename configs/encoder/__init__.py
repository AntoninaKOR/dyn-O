from dataclasses import dataclass, field

from configs.encoder.solv_sam import SolvSamConfig


@dataclass
class EncoderConfig:
    solv_sam: SolvSamConfig = field(default_factory=SolvSamConfig)

    def __post_init__(self):
        self.configs = {
            "solv_sam": self.solv_sam,
        }
