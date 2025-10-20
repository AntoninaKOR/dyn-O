import dataclasses
from typing import Tuple, Dict
from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

from trainer.world_model_env import WorldModelEnv
from policy.bc.bc_model import BehaviorCloningModel
from oc_utils.utils import REPO_PATH, are_dicts_equal
from oc_utils.replay_buffer.data_format import Episode


class BehaviorCloning(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = BehaviorCloningModel(config)

        self.load_checkpoint(config.policy.checkpoint_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.policy.training.lr)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, *args, **kwargs):
        if self.training:
            return self.compute_loss(*args, **kwargs)
        else:
            return self.act(*args, **kwargs)

    def act(
        self,
        obs: torch.FloatTensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:

        logits_actions = self.model(obs)

        if deterministic:
            act = logits_actions.argmax(dim=-1)
        else:
            act = Categorical(logits=logits_actions / temperature).sample()

        return act

    def compute_loss(
        self,
        batch: Episode,
        world_model_env: WorldModelEnv,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        obs = world_model_env.get_obs(batch)
        logits_actions = self.model(obs)
        logits_actions = logits_actions.view(-1, logits_actions.size(-1))
        actions = batch.actions.view(-1).to(torch.int64)
        loss = F.cross_entropy(logits_actions, actions)
        accuracy = (actions == logits_actions.argmax(dim=-1)).float().mean()

        logging = {
            "loss": loss,
            "accuracy": accuracy,
        }

        return loss, logging

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
            keys=["model"],
        ), "policy_config mismatch (see above)"
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
            "encoder_config": dataclasses.asdict(self.config.encoder),
            "policy_config": dataclasses.asdict(self.config.policy),
        }
