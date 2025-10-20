import dataclasses

import torch
import torch.nn as nn

from configs.config import Config
from oc_utils.layers.mlp import mlp
from oc_utils.layers.transformer_block import Transformer
from trainer.world_model_env import WorldModelObs


class BehaviorCloningModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.policy.model

        d_model = self.model_config.mha_cfg.d_model

        # projection layer: input dim to d_model
        if self.model_config.input_mode == "enc_feat":
            input_dim = config.encoder.token_dim
        elif self.model_config.input_mode == "slots":
            input_dim = config.encoder.slot_dim
        else:
            raise ValueError(f"Unknown input_mode: {self.model_config.input_mode}")

        self.proj_in = mlp(
            input_dim,
            self.model_config.proj_in_dim,
            d_model,
        )

        self.action_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.action_token, std=1e-6)

        mha_cfg = dataclasses.asdict(self.model_config.mha_cfg)
        self.mha = Transformer(**mha_cfg)

        self.proj_out = nn.Linear(d_model, config.action_space.n)

    def forward(self, obs: WorldModelObs, **kwargs) -> torch.Tensor:
        # obs_encodings: (bs, num_tokens, 768) or (bs, num_slots, slot_dim)
        if self.model_config.input_mode == "enc_feat":
            obs_encodings = obs.enc_feat
        elif self.model_config.input_mode == "slots":
            obs_encodings = obs.slots.clone()
            obs_encodings[~obs.slots_exist] = 0.
        else:
            raise ValueError(f"Unknown input_mode: {self.model_config.input_mode}")

        ori_bs = obs_encodings.shape[:-2]
        obs_encodings = obs_encodings.view(-1, *obs_encodings.shape[-2:])
        obs_encodings = obs_encodings.clone()

        x = self.proj_in(obs_encodings)                                                 # (bs, num_tokens, d_model)
        x = torch.cat([
            self.action_token.expand(x.shape[0], -1, -1),
            x,
        ], dim=1)                                                                       # (bs, num_tokens + 1, d_model)
        x = self.mha(x)                                                                 # (bs, num_tokens + 1, d_model)
        x = x[:, 0, :]                                                                  # (bs, d_model)
        x = self.proj_out(x).view(*ori_bs, -1)                                          # (bs, num_actions)

        return x

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
