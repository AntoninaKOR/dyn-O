from policy.actor_critic.actor_critic import ActorCritic
from policy.bc.bc import BehaviorCloning
from policy.sac import SAC

policy_library = {
    "sac": SAC,
    "actor_critic": ActorCritic,
    "bc": BehaviorCloning,
}
