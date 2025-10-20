import dataclasses


@dataclasses.dataclass
class SacConfig:
    # === Parameters ===
    gamma: float = 0.99
    tau: float = 1.0
    alpha: float = 0.2
    autotune: bool = True
    target_entropy_scale: float = 0.89

    # === Training ===
    learning_starts: int = 20000
    q_lr: float = 0.0003
    policy_lr: float = 0.0003
    batch_size: int = 64
    update_frequency: int = 4
    target_network_frequency: int = 8000
