import dataclasses


@dataclasses.dataclass
class OnlineConfig:
    num_envs: int = 40
    buffer_size: int = 1000000
