from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange

from oc_utils.utils import FloatTensor
from oc_utils.replay_buffer.data_format import Episode
from dynamics.mamba.generation import Prediction


@dataclass
class WorldModelObs:
    slots: Optional[FloatTensor] = None
    slots_exist: Optional[torch.BoolTensor] = None
    enc_feat: Optional[FloatTensor] = None


class WorldModelEnv:
    def __init__(self, config, encoder, dynamics) -> None:
        self.config = config
        self.world_model_env_config = config.policy.world_model_env

        self.dynamics = dynamics
        self.encoder = encoder

        self.step_rollout = None

    @torch.no_grad()
    def reset(
        self,
        batch: Dict[str, Union[torch.Tensor, None]],
    ) -> WorldModelObs:
        # slots: (b, n_warmup_steps, n, d)
        # act: (b, n_warmup_steps)
        self.dynamics.eval()
        self.encoder.eval()
        if self.world_model_env_config.random_static:
            batch = self.dynamics.perturb_static(
                batch,
                self.world_model_env_config.random_one_slot,
                self.world_model_env_config.random_prob,
            )
        self.step_rollout = self.dynamics.reset_rollout(
            batch,
            self.world_model_env_config.n_rollout_steps,
            self.world_model_env_config.deterministic,
        )

        return self.get_obs(self.step_rollout)

    @torch.no_grad()
    def step(
        self,
        action: Union[int, np.ndarray, torch.LongTensor],
    ) -> Tuple[WorldModelObs, FloatTensor, torch.BoolTensor, None]:

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long).reshape(-1).to(self.dynamics.device)

        self.dynamics.eval()
        self.encoder.eval()
        action = rearrange(action, "b -> b 1")
        self.step_rollout = self.dynamics.step_rollout(
            self.step_rollout,
            action,
            self.world_model_env_config.deterministic,
        )

        reward = rearrange(self.step_rollout.rewards_pred, "b 1 -> b")
        termination = rearrange(self.step_rollout.terminations_pred, "b 1 -> b")

        return self.get_obs(self.step_rollout), reward, termination, None

    @torch.no_grad()
    def get_obs(self, step_rollout: Episode | Prediction) -> WorldModelObs:

        if isinstance(step_rollout, Episode):
            slots = step_rollout.slots
            slots_exist = step_rollout.slots_exist
            slots_visible = step_rollout.slots_visible
        elif isinstance(step_rollout, Prediction):
            slots = step_rollout.next_slots_pred
            slots_exist = step_rollout.next_slots_exist_pred
            slots_visible = step_rollout.next_slots_visible_pred
        else:
            raise ValueError(f"Invalid step_rollout type: {type(step_rollout)}")

        if self.world_model_env_config.obs_mode == "enc_feat":
            if isinstance(step_rollout, Episode):
                enc_feat = step_rollout.enc_feat
            elif isinstance(step_rollout, Prediction):
                if self.config.dynamics.patch_as_slot:
                    enc_feat = slots
                else:
                    enc_feat = self.encoder.decode(slots, slots_visible, modes="enc_feat_rec")
            else:
                enc_feat = None
        elif self.world_model_env_config.obs_mode == "decode_rgb_then_enc_feat":
            image_rec = self.encoder.decode(slots, slots_visible, modes="rgb_rec")
            enc_feat = self.encoder(image_rec, mode="enc_feat")
        else:
            enc_feat = None

        if slots is not None and slots.shape[1] == 1:
            slots = rearrange(slots, "b 1 n d -> b n d")
        if slots_exist is not None and slots_exist.shape[1] == 1:
            slots_exist = rearrange(slots_exist, "b 1 n -> b n")
        if enc_feat is not None and enc_feat.shape[1] == 1:
            enc_feat = rearrange(enc_feat, "b 1 n d -> b n d")

        obs = WorldModelObs(
            slots=slots,
            slots_exist=slots_exist,
            enc_feat=enc_feat,
        )

        return obs
