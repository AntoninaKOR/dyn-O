import numpy as np

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities import Metric
from mlflow.utils.time import get_current_time_millis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_shape = config.encoding_space.shape
        self.fc1 = layer_init(nn.Linear(np.prod(self.obs_shape), 512))
        self.fc_q = layer_init(nn.Linear(512, config.action_space.n))

    def forward(self, x):
        x = x.view(*x.shape[:-len(self.obs_shape)], -1)
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_shape = config.encoding_space.shape
        self.fc1 = layer_init(nn.Linear(np.prod(self.obs_shape), 512))
        self.fc_logits = layer_init(nn.Linear(512, config.action_space.n))

    def forward(self, x):
        x = x.view(*x.shape[:-len(self.obs_shape)], -1)
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class SAC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.actor = Actor(config).to(config.device)
        self.qf1 = SoftQNetwork(config).to(config.device)
        self.qf2 = SoftQNetwork(config).to(config.device)
        self.qf1_target = SoftQNetwork(config).to(config.device)
        self.qf2_target = SoftQNetwork(config).to(config.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=config.policy.q_lr,
            eps=1e-4,
        )
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=config.policy.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        if config.policy.autotune:
            self.target_entropy = -config.policy.target_entropy_scale * torch.log(1 / torch.tensor(config.action_space.n))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=config.policy.q_lr, eps=1e-4)
        else:
            self.alpha = config.policy.self.alpha

    def get_action(self, obs):
        return self.actor.get_action(obs)

    def update(self, replay_buffer, global_step):
        policy_config = self.config.policy
        encoding, actions, rewards, dones, next_encoding = replay_buffer.sample(policy_config.batch_size)
        encoding, actions, rewards, dones, next_encoding = map(
            lambda x: (
                torch.tensor(x, dtype=self.config.dtype, device=self.config.device)
                if isinstance(x, np.ndarray)
                else x.to(dtype=self.config.dtype, device=self.config.device)
            ),
            [encoding, actions, rewards, dones, next_encoding]
        )
        actions = actions.unsqueeze(1)          # (bs, ) -> (bs, 1)

        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(next_encoding)
            qf1_next_target = self.qf1_target(next_encoding)
            qf2_next_target = self.qf2_target(next_encoding)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * policy_config.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values = self.qf1(encoding)
        qf2_values = self.qf2(encoding)
        qf1_a_values = qf1_values.gather(1, actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(encoding)
        with torch.no_grad():
            qf1_values = self.qf1(encoding)
            qf2_values = self.qf2(encoding)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if policy_config.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update the target networks

        if global_step % policy_config.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(policy_config.tau * param.data + (1 - policy_config.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(policy_config.tau * param.data + (1 - policy_config.tau) * target_param.data)

        if global_step % 100 == 0:
            ts = get_current_time_millis()
            metrics = [
                Metric("policy/qf1_values", qf1_a_values.mean().item(), ts, global_step),
                Metric("policy/qf2_values", qf2_a_values.mean().item(), ts, global_step),
                Metric("policy/qf1_loss", qf1_loss.item(), ts, global_step),
                Metric("policy/qf2_loss", qf2_loss.item(), ts, global_step),
                Metric("policy/qf_loss", qf_loss.item() / 2.0, ts, global_step),
                Metric("policy/actor_loss", actor_loss.item(), ts, global_step),
                Metric("policy/self.alpha", self.alpha, ts, global_step),
            ]
            if policy_config.autotune:
                metrics.append(Metric("policy/alpha_loss", alpha_loss.item(), ts, global_step))
            MlflowClient().log_batch(mlflow.active_run().info.run_id, metrics)
