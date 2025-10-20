from dataclasses import dataclass, field

from configs.policy.sac import SacConfig
from configs.policy.actor_critic import ActorCriticConfig
from configs.policy.bc import BehaviorCloningConfig


@dataclass
class PolicyConfig:
    sac: SacConfig = field(default_factory=SacConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    bc: BehaviorCloningConfig = field(default_factory=BehaviorCloningConfig)

    def __post_init__(self):
        self.configs = {
            "sac": self.sac,
            "actor_critic": self.actor_critic,
            "bc": self.bc,
        }
