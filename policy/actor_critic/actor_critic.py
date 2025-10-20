import numpy as np
import dataclasses
from copy import deepcopy
from typing import Tuple, Dict, Union
from pathlib import Path
from einops import rearrange

import torch
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.nn.functional as F

from trainer.world_model_env import WorldModelEnv
from policy.actor_critic.actor_critic_model import ActorCriticModel
from policy.actor_critic.utils import compute_lambda_returns, compute_mask_after_first_done
from oc_utils.reward import symexp, symlog, two_hot, compute_softmax_over_buckets
from oc_utils.utils import REPO_PATH, are_dicts_equal, torch_cat_dataclasses_list
from oc_utils.replay_buffer.data_format import Episode


@dataclasses.dataclass
class ImagineOutput:
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    logits_values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    target_values: torch.FloatTensor
    value_bootstrap: torch.FloatTensor


class ActorCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.use_imagination = config.policy.training.use_imagination
        self.two_hot_rets = config.policy.actor_critic_model.two_hot_rets
        self.model = ActorCriticModel(config)
        self.target_model = deepcopy(self.model)
        self.target_model.requires_grad_(False)

        self.load_checkpoint(config.policy.checkpoint_path)

        self.should_use_imag_accumulator = 0.0

    def configure_optimizers(self):
        return {
            "actor": torch.optim.Adam(
                self.model.actor_parameters,
                lr=self.config.policy.training.actor_lr,
                eps=self.config.policy.training.eps,
            ),
            "critic": torch.optim.Adam(
                self.model.critic_parameters,
                lr=self.config.policy.training.critic_lr,
                eps=self.config.policy.training.eps,
            ),
        }

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, *args, **kwargs):
        if self.training:
            return self.compute_loss(*args, **kwargs)
        else:
            return self.act(*args, **kwargs)

    def update_target(self) -> None:
        TAU = self.config.policy.training.target_update_tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def act(
        self,
        obs: torch.FloatTensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:

        outputs = self.model(obs, act_only=True)

        if deterministic:
            act = outputs.logits_actions.argmax(dim=-1)
        else:
            act = Categorical(logits=outputs.logits_actions / temperature).sample()

        return act

    def compute_loss(
        self,
        batch: Episode,
        world_model_env: WorldModelEnv,
        **kwargs,
    ) -> Tuple[Dict[str, torch.FloatTensor], Dict[str, float | torch.FloatTensor]]:
        self.update_target()

        self.should_use_imag_accumulator += self.config.policy.training.imagination_prob
        if self.use_imagination and self.should_use_imag_accumulator >= 1.0:
            self.should_use_imag_accumulator -= 1.0
            batch_size = batch.actions.shape[0]
            minibatch_size = min(batch_size, self.config.policy.training.minibatch_size)
            outputs = []
            for i in range(0, batch_size, minibatch_size):
                batch_i = Episode(
                    slots=None if batch.slots is None else batch.slots[i:i + minibatch_size],
                    enc_feat=None if batch.enc_feat is None else batch.enc_feat[i:i + minibatch_size],
                    slots_visible=batch.slots_visible[i:i + minibatch_size],
                    slots_exist=batch.slots_exist[i:i + minibatch_size],
                    actions=batch.actions[i:i + minibatch_size],
                    rewards=batch.rewards[i:i + minibatch_size],
                    terminations=batch.terminations[i:i + minibatch_size],
                    truncations=batch.truncations[i:i + minibatch_size],
                )
                output = self.imagine(batch_i, world_model_env)
                outputs.append(output)
            outputs = torch_cat_dataclasses_list(outputs, dim=0)
        else:
            outputs = self.fake_imagine(batch, world_model_env)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.target_values,
                ends=outputs.ends,
                value_bootstrap=outputs.value_bootstrap,
                gamma=self.config.policy.training.gamma,
                lambda_=self.config.policy.training.lambda_,
            )

        mask = compute_mask_after_first_done(outputs.ends)

        lambda_returns = lambda_returns[mask]
        target_values = outputs.target_values[mask]
        logits_values = outputs.logits_values[mask]
        logits_actions = outputs.logits_actions[mask]
        actions = outputs.actions[mask]

        d = Categorical(logits=logits_actions)
        log_probs = d.log_prob(actions)

        adv = lambda_returns - target_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = (log_probs - log_probs.detach()).exp()
        loss_actions = torch.mean(-ratio * adv)

        if self.two_hot_rets:
            pred_values = symexp(compute_softmax_over_buckets(logits_values))
            loss_values = F.cross_entropy(logits_values, target=two_hot(symlog(lambda_returns)))
        else:
            pred_values = logits_values
            loss_values = F.mse_loss(logits_values, lambda_returns)

        entropy = torch.mean(d.entropy())
        # loss_entropy = -self.config.policy.training.entropy_weight * entropy

        kl = kl_divergence(Categorical(logits=torch.zeros_like(logits_actions)), d)
        max_prob = d.probs.max(dim=-1).values
        apply_kl_reg_mask = max_prob > 1 - self.config.policy.training.action_uniform_prob
        masked_kl = kl[apply_kl_reg_mask].mean()
        loss_entropy = self.config.policy.training.entropy_weight * masked_kl if torch.any(apply_kl_reg_mask) else 0

        loss = {
            "actor": loss_actions + loss_entropy,
            "critic": loss_values,
        }
        logging = {
            "loss": loss_actions + loss_values + loss_entropy,
            "loss_actions": loss_actions,
            "loss_values": loss_values,
            "loss_entropy": loss_entropy,
            "policy_entropy": entropy,
            "policy_kl": kl.mean(),
            "max_prob": max_prob.mean(),
            "value": pred_values.mean(),
            "return": lambda_returns.mean(),
            "target_value": target_values.mean(),
            "advantage": adv.mean(),
        }

        return loss, logging

    def uni_mix_categorical(self, logits):
        probs = F.softmax(logits, dim=-1)
        max_prob, max_prob_indices = probs.max(dim=-1)
        max_onehot = F.one_hot(max_prob_indices, num_classes=self.config.action_space.n)

        uniform_prob = self.config.policy.training.action_uniform_prob
        non_max_probs = 1 - max_prob
        uniform_prob = torch.clip(uniform_prob - non_max_probs, 0)
        uniform_prob = rearrange(uniform_prob, '... -> ... 1')

        prob_change = -uniform_prob * max_onehot + uniform_prob * (1 - max_onehot) / (self.config.action_space.n - 1)

        probs = probs + prob_change
        d = Categorical(probs=probs)
        return d

    def imagine(self, batch: Episode, world_model_env: WorldModelEnv) -> ImagineOutput:
        assert batch.slots.shape[1] == self.config.dynamics.rollout.n_warmup_steps

        all_actions = []
        all_logits_actions = []
        all_logits_values = []
        all_target_logits_values = []
        all_rewards = []
        all_ends = []

        obs = world_model_env.reset(batch)
        for _ in range(self.config.policy.world_model_env.n_rollout_steps):
            outputs = self.model(obs)
            action = self.uni_mix_categorical(outputs.logits_actions).sample()              # (B, A)

            with torch.no_grad():
                target_logits_values = self.target_model(obs).logits_values                 # (B, 255) or (B, 1)
                obs, reward, terminate, _ = world_model_env.step(action)

            all_actions.append(action)
            all_logits_actions.append(outputs.logits_actions)
            all_logits_values.append(outputs.logits_values if self.two_hot_rets else outputs.logits_values[:, 0])
            all_target_logits_values.append(target_logits_values if self.two_hot_rets else target_logits_values[:, 0])
            all_rewards.append(reward)
            all_ends.append(terminate)

        with torch.no_grad():
            logits_values = self.target_model(obs).logits_values
            if self.two_hot_rets:
                value_bootstrap = symexp(compute_softmax_over_buckets(logits_values))
            else:
                value_bootstrap = logits_values[:, 0]

            target_values = torch.stack(all_target_logits_values, dim=1)
            if self.two_hot_rets:
                target_values = symexp(compute_softmax_over_buckets(target_values))

        return ImagineOutput(
            actions=torch.stack(all_actions, dim=1),                                        # (B, T)
            logits_actions=torch.stack(all_logits_actions, dim=1),                          # (B, T, A)
            logits_values=torch.stack(all_logits_values, dim=1),                            # (B, T, 255)
            rewards=torch.stack(all_rewards, dim=1),                                        # (B, T)
            ends=torch.stack(all_ends, dim=1).long(),                                       # (B, T)
            target_values=target_values,                                                    # (B, T, 255)
            value_bootstrap=value_bootstrap,                                                # (B,)
        )

    def fake_imagine(self, batch: Episode, world_model_env: WorldModelEnv) -> ImagineOutput:
        # assert batch.slots.shape[1] == self.config.policy.world_model_env.n_rollout_steps + 1

        obs = world_model_env.get_obs(batch)

        outputs = self.model(obs)

        all_logits_actions = outputs.logits_actions[:, :-1]                                 # (B, T, A)
        all_logits_values = outputs.logits_values if self.two_hot_rets else outputs.logits_values[..., 0]
        all_logits_values = all_logits_values[:, :-1]                                       # (B, T, 255) or (B, T)

        target_outputs = self.target_model(obs)
        target_logits_values = target_outputs.logits_values                                 # (B, 255) or (B, 1)
        all_target_logits_values = target_logits_values if self.two_hot_rets else target_logits_values[..., 0]
        all_target_logits_values = all_target_logits_values[:, :-1]                         # (B, T, 255) or (B, T)

        all_actions = batch.actions[:, :-1]                                                 # (B, A)
        all_rewards = batch.rewards[:, :-1]
        all_ends = batch.terminations[:, :-1]

        with torch.no_grad():
            logits_values = target_outputs.logits_values[:, -1]
            if self.two_hot_rets:
                value_bootstrap = symexp(compute_softmax_over_buckets(logits_values))
            else:
                value_bootstrap = logits_values[:, 0]

            if self.two_hot_rets:
                target_values = symexp(compute_softmax_over_buckets(all_target_logits_values))

        return ImagineOutput(
            actions=all_actions,                                                            # (B, T)
            logits_actions=all_logits_actions,                                              # (B, T, A)
            logits_values=all_logits_values,                                                # (B, T, 255)
            rewards=all_rewards,                                                            # (B, T)
            ends=all_ends.long(),                                                           # (B, T)
            target_values=target_values,                                                    # (B, T, 255)
            value_bootstrap=value_bootstrap,                                                # (B,)
        )

    # ============ Save and Load ============
    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_PATH / checkpoint_path
        assert checkpoint_path.exists(), f"=> no policy checkpoint found at '{checkpoint_path}'"

        # open checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # verify config is the same as encoder training
        assert are_dicts_equal(
            dataclasses.asdict(self.config.policy),
            checkpoint["policy_config"],
            keys=["actor_critic_model"],
        ), "policy_config mismatch (see above)"
        assert are_dicts_equal(
            dataclasses.asdict(self.config.dynamics),
            checkpoint["dynamics_config"],
            keys=["num_slots"],
        ), "dynamics_config mismatch (see above)"
        assert are_dicts_equal(
            dataclasses.asdict(self.config.encoder),
            checkpoint["encoder_config"],
            keys=["slot_dim", "token_num"],
        ), "encoder_config mismatch (see above)"

        # load model state dict
        model_state_dict = checkpoint["model"]

        # remove ddp prefix
        model_state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in model_state_dict.items()
        }
        msg = self.load_state_dict(model_state_dict, strict=True)
        print(f"=> policy loaded model from checkpoint: {checkpoint_path} with msg {msg}")

    def save_dict(self):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        return {
            "model": self.state_dict(),
            "dynamics_config": dataclasses.asdict(self.config.dynamics),
            "encoder_config": dataclasses.asdict(self.config.encoder),
            "policy_config": dataclasses.asdict(self.config.policy),
        }
