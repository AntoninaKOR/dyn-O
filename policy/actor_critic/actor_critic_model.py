import dataclasses

import torch
import torch.nn as nn
from einops import rearrange

from configs.config import Config
from oc_utils.layers.mlp import mlp
from oc_utils.layers.transformer_block import Transformer
from trainer.world_model_env import WorldModelObs


@dataclasses.dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    logits_values: torch.FloatTensor | None


class ActorCriticModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.ac_model_config = config.policy.actor_critic_model
        self.downsample_with_conv = self.ac_model_config.downsample_with_conv

        d_model = self.ac_model_config.mha_cfg.d_model

        # projection layer: input dim to d_model
        if self.ac_model_config.input_mode == "enc_feat":
            input_dim = config.encoder.token_dim
        elif self.ac_model_config.input_mode == "slots":
            input_dim = config.encoder.slot_dim
        else:
            raise ValueError(f"Unknown input_mode: {self.ac_model_config.input_mode}")

        if self.downsample_with_conv:
            assert self.ac_model_config.input_mode == "enc_feat"
            self.actor_cnn = nn.Conv2d(input_dim, input_dim, kernel_size=4, stride=4)
            self.critic_cnn = nn.Conv2d(input_dim, input_dim, kernel_size=4, stride=4)
        else:
            self.actor_cnn = self.critic_cnn = None

        self.actor_proj_in = mlp(
            input_dim,
            self.ac_model_config.proj_in_dim,
            d_model,
        )
        self.critic_proj_in = mlp(
            input_dim,
            self.ac_model_config.proj_in_dim,
            d_model,
        )

        # actor and critic embeddings
        self.actor_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.critic_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.actor_token, std=1e-6)
        nn.init.normal_(self.critic_token, std=1e-6)

        # self-attention blocks
        mha_cfg = dataclasses.asdict(self.ac_model_config.mha_cfg)
        self.actor_mha = Transformer(**mha_cfg)
        self.critic_mha = Transformer(**mha_cfg)

        # actor and critic heads
        self.actor_mlp = nn.Linear(d_model, config.action_space.n)
        nn.init.constant_(self.actor_mlp.bias.data, 0)

        self.critic_mlp = nn.Linear(d_model, 255 if self.ac_model_config.two_hot_rets else 1)
        nn.init.constant_(self.critic_mlp.weight.data, 0)
        nn.init.constant_(self.critic_mlp.bias.data, 0)

    @property
    def actor_parameters(self):
        return list(self.actor_proj_in.parameters()) + \
            [self.actor_token] + \
            list(self.actor_mha.parameters()) + \
            list(self.actor_mlp.parameters())

    @property
    def critic_parameters(self):
        return list(self.critic_proj_in.parameters()) + \
            [self.critic_token] + \
            list(self.critic_mha.parameters()) + \
            list(self.critic_mlp.parameters())

    def forward(self, obs: WorldModelObs, act_only: bool = False, **kwargs) -> ActorCriticOutput:
        # obs_encodings: (bs, num_tokens, 768) or (bs, num_slots, slot_dim)
        if self.ac_model_config.input_mode == "enc_feat":
            obs_encodings = obs.enc_feat
        elif self.ac_model_config.input_mode == "slots":
            obs_encodings = obs.slots.clone()
            if obs.slots_exist is not None:
                obs_encodings[~obs.slots_exist] = 0.
        else:
            raise ValueError(f"Unknown input_mode: {self.ac_model_config.input_mode}")

        ori_bs = obs_encodings.shape[:-2]
        obs_encodings = obs_encodings.view(-1, *obs_encodings.shape[-2:])
        obs_encodings = obs_encodings.clone()

        outputs = []
        modules = [
            [self.actor_cnn, self.actor_proj_in, self.actor_token, self.actor_mha, self.actor_mlp],
        ]
        if not act_only:
            modules.append([self.critic_cnn, self.critic_proj_in, self.critic_token, self.critic_mha, self.critic_mlp])

        for cnn, proj_in, token, mha, post_mlp in modules:
            if self.downsample_with_conv:
                h, w = self.config.encoder.resize_to
                h_p = h // self.config.encoder.patch_size
                w_p = w // self.config.encoder.patch_size
                x = rearrange(obs_encodings, '... (h_p w_p) d -> ... d h_p w_p', h_p=h_p, w_p=w_p)
                x = cnn(x)                                                          # (bs, d_model, h_p // 2, w_p // 2)
                x = rearrange(x, '... d h_p w_p -> ... (h_p w_p) d')
                x = nn.functional.relu(x)
            else:
                x = obs_encodings

            x = proj_in(x)                                                          # (bs, num_tokens, d_model)
            x = torch.cat([
                token.expand(x.shape[0], -1, -1),
                x,
            ], dim=1)                                                               # (bs, num_tokens + 1, d_model)
            x = mha(x)                                                              # (bs, num_tokens + 1, d_model)
            x = x[:, 0, :]                                                          # (bs, d_model)
            x = post_mlp(x).view(*ori_bs, -1)                                       # (bs, num_actions)
            outputs.append(x)

        if act_only:
            logits_actions = outputs[0]
            logits_values = None
        else:
            logits_actions, logits_values = outputs

        return ActorCriticOutput(logits_actions, logits_values)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
