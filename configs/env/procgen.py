import dataclasses


@dataclasses.dataclass
class ProcgenConfig:
    env_id: str = "coinrun"
    distribution_mode: str = "hard"
    use_backgrounds: bool = False

    num_train_envs: int = 10
    num_eval_envs: int = 10
