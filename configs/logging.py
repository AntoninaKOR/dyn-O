from typing import List
from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    training_log_interval: int = 20
    ckpt_per_timestep: int = int(1e5)
    ckpt_per_second: int = 600

    # dynamics rollout visualization
    viz_rollout_per_timestep: int = 5000
    num_viz_rollouts: int = 2
    rollout_viz_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])

    # policy gif visualization
    eval_policy_per_timestep: int = 2500
    num_of_gifs: int = 5
