# modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
import copy
import dataclasses
import gymnasium
import h5py
import numpy as np

from collections import OrderedDict
from dataclasses import fields
from einops import rearrange, repeat
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from dynamics.mamba.mamba_mixer import MixerModel
from dynamics.mamba.generation import GenerationMixin, Prediction
from dynamics.mamba.kmeans import KMeans

from dynamics.mamba.quantization.fsq import FiniteScalarQuantization
from dynamics.mamba.quantization.sim_vq import SimpleVectorQuantization
from dynamics.mamba.quantization.vq import VectorQuantization
from dynamics.mamba.utils import log_sinkhorn_algorithm, LeCAM_EMA

from oc_utils.layers.transformer_block import Transformer
from oc_utils.layers.mlp import mlp

from oc_utils.replay_buffer.data_format import Episode
from oc_utils.reward import symexp, symlog, two_hot, compute_softmax_over_buckets
from oc_utils.utils import REPO_PATH, are_dicts_equal, torch_cat_dataclasses_list


class MambaWorldModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: Config,
        encoder,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.factory_kwargs = {"device": config.device, "dtype": config.dtype}

        # ============ MambaConfig ============
        d_model = config.dynamics.mamba.d_model
        n_layer = config.dynamics.mamba.n_layer
        d_intermediate = config.dynamics.mamba.d_intermediate

        layer = config.dynamics.mamba.layer
        attn_cfg = config.dynamics.mamba.attn_cfg
        ssm_cfg = config.dynamics.mamba.ssm_cfg

        assert layer in ["Mamba1", "Mamba2", "MHA"], f"Invalid layer type: {layer}"
        if layer == "MHA":
            attn_layer_idx = list(range(n_layer))           # all layers use attention instead of mamba
        else:
            attn_layer_idx = []                             # no layer uses attention
            ssm_cfg = dataclasses.asdict(ssm_cfg)
            ssm_cfg["layer"] = layer

        rms_norm = config.dynamics.mamba.rms_norm
        residual_in_fp32 = config.dynamics.mamba.residual_in_fp32
        fused_add_norm = config.dynamics.mamba.fused_add_norm

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=None,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **self.factory_kwargs,
        )

        # ============ Miscs ============
        self.num_slots = config.dynamics.num_slots
        self.disentangle_static_dynamic = config.dynamics.pred.disentangle_static_dynamic
        self.static_dynamic_merge_method = config.dynamics.pred.static_dynamic_merge_method

        if config.dynamics.patch_as_slot:
            self.slot_dim = config.encoder.token_dim
            assert not self.disentangle_static_dynamic, "patch_as_slot is not compatible with disentangle_static_dynamic"
        else:
            self.slot_dim = config.encoder.slot_dim

        if config.dynamics.training.slot_loss_fn == "mse":
            self.slot_loss_fn = F.mse_loss
        elif config.dynamics.training.slot_loss_fn == "l1":
            self.slot_loss_fn = F.l1_loss
        else:
            raise ValueError(f"Invalid loss function: {config.dynamics.training.slot_loss_fn}")

        self.use_slot_visible_and_exists = self.config.encoder.use_sam_mask

        # ============ K-means for sampling new features ============
        self.slot_kmeans = self.load_slot_kmeans()

        # ============ Slot -> Embedding ============

        # if disentangle_static_dynamic:
        #   slot -> static subpart + dynamic subpart
        #   dynamic subpart (+ static subpart + visible + exists) -> embedding
        # else:
        #   slot (+ visible + exists) -> embedding
        if self.disentangle_static_dynamic:
            self.proj_static = mlp(
                self.slot_dim,
                config.dynamics.pred.proj_mlp_dims,
                self.slot_dim,
                norm=config.dynamics.pred.mlp_use_norm,
            )
            self.proj_dynamic = mlp(
                self.slot_dim,
                config.dynamics.pred.proj_mlp_dims,
                self.slot_dim,
                norm=config.dynamics.pred.mlp_use_norm,
            )

            if config.dynamics.pred.dynamic_use_vq:
                if config.dynamics.pred.dynamic_vq_type == "vq":
                    self.dynamic_vq = VectorQuantization(
                        codebook_size=config.dynamics.pred.codebook_size,
                        codebook_dim=self.slot_dim,
                        commit_weight=config.dynamics.training.commit_weight,
                        codebook_weight=config.dynamics.training.input_to_quantize_commit_loss_weight,
                        ema_update=config.dynamics.training.vq_ema_update,
                    )
                elif config.dynamics.pred.dynamic_vq_type == "sim_vq":
                    self.dynamic_vq = SimpleVectorQuantization(
                        codebook_size=config.dynamics.pred.codebook_size,
                        codebook_dim=self.slot_dim,
                        codebook_transform_mlp_multi=config.dynamics.pred.codebook_transform_mlp_multi,
                        commit_weight=config.dynamics.training.commit_weight,
                        input_to_quantize_commit_loss_weight=config.dynamics.training.input_to_quantize_commit_loss_weight,
                    )
                elif config.dynamics.pred.dynamic_vq_type == "fsq":
                    self.dynamic_vq = FiniteScalarQuantization(
                        codebook_levels=config.dynamics.pred.codebook_levels,
                        codebook_dim=self.slot_dim,
                    )
                else:
                    raise ValueError(f"Invalid VQ type: {config.dynamics.pred.dynamic_vq_type}")

            # static subpart + dynamic subpart -> slot
            assert self.static_dynamic_merge_method in ["concat", "add", "dynamic_only"]
            self.slot_recon = mlp(
                2 * self.slot_dim if self.static_dynamic_merge_method == "concat" else self.slot_dim,
                config.dynamics.pred.recon_mlp_dims,
                self.slot_dim,
                norm=config.dynamics.pred.mlp_use_norm,
            )

            # dynamic subpart -> static subpart, to minimize the mutual information between them
            self.dynamic_to_static_discriminator = mlp(
                2 * self.slot_dim,
                config.dynamics.pred.dynamic_to_static_discriminator_mlp_dims,
                1,
                norm=config.dynamics.pred.mlp_use_norm,
            )

            self.discriminator_regularization_type = config.dynamics.training.discriminator_regularization_type
            if self.discriminator_regularization_type == "lecam":
                self.lecam_ema = LeCAM_EMA()

        proj_in_dim = self.slot_dim
        if self.use_slot_visible_and_exists:
            proj_in_dim += 2

        self.proj_in = mlp(
            proj_in_dim,
            config.dynamics.pred.proj_in_mlp_dims,
            d_model,
            norm=config.dynamics.pred.mlp_use_norm,
        )

        # ============ Action to Embedding ============
        if isinstance(config.action_space, gymnasium.spaces.Discrete):
            num_actions = config.action_space.n
            self.action_embedding = nn.Embedding(num_actions, d_model, **self.factory_kwargs)
        elif isinstance(config.action_space, gymnasium.spaces.Box):
            assert len(config.action_space.shape) == 1, "Action space must be 1D"
            self.action_proj_in = mlp(
                config.action_space.shape[0],
                config.dynamics.pred.proj_in_mlp_dims,
                d_model,
                norm=config.dynamics.pred.mlp_use_norm,
            )
        else:
            raise ValueError(f"Unsupported action space type: {type(config.action_space)}")

        # ============ Slot Embedding Mixer before Passing into SSM ============
        pre_ssm_mixer_cfg = dataclasses.asdict(config.dynamics.pred.pre_ssm_mixer_cfg)
        self.pre_ssm_mixer = Transformer(d_model=d_model, **pre_ssm_mixer_cfg)

        # ============ Next-Step Slot (value, visible, exists) Prediction ============
        self.num_slot_modes = config.dynamics.pred.num_slot_modes

        # multi-modal prediction for each slot
        slot_pred_out_dim = self.num_slot_modes * (self.slot_dim + 1)
        if self.use_slot_visible_and_exists:
            slot_pred_out_dim += 2
        self.slot_predictor = mlp(d_model, config.dynamics.pred.pred_mlp_dims, slot_pred_out_dim, norm=config.dynamics.pred.mlp_use_norm)

        # ============ Reward and Termination Prediction ============
        pred_mha_cfg = dataclasses.asdict(config.dynamics.pred.pred_mha_cfg)

        self.reward_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.reward_token, std=1e-6)
        self.reward_pred_mha = Transformer(d_model=d_model, **pred_mha_cfg)
        self.reward_pred_mlp = nn.Linear(d_model, config.dynamics.pred.num_reward_bins)

        self.termination_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.termination_token, std=1e-6)
        self.termination_pred_mha = Transformer(d_model=d_model, **pred_mha_cfg)
        self.termination_pred_mlp = nn.Linear(d_model, 1)

        self.encoder = encoder

        # for loss term annealing
        self.step = 0
        self.load_checkpoint(config.dynamics.checkpoint_path)

    def shared_modules(self):
        return {
            "proj_dynamic": [self.proj_dynamic],
        }

    def zero_grad_shared_modules(self):
        for module_group in self.shared_modules().values():
            for mm in module_group:
                mm.zero_grad()

    def configure_optimizers(self):
        encoder_params = list(self.encoder.parameters())
        encoder_params_ids = [id(param) for param in encoder_params]

        if self.disentangle_static_dynamic:

            static_params = list(self.proj_static.parameters())
            dynamic_params = list(self.proj_dynamic.parameters()) + list(self.slot_recon.parameters())
            if self.config.dynamics.pred.dynamic_use_vq:
                dynamic_params += list(self.dynamic_vq.parameters())
            discriminator_params = list(self.dynamic_to_static_discriminator.parameters())

            exclude_params_ids = [id(param) for param in static_params + dynamic_params + discriminator_params + encoder_params]
            other_params = [
                param for param in self.parameters()
                if id(param) not in exclude_params_ids
            ]

            optimizers = [
                ("static", torch.optim.AdamW(static_params, lr=self.config.dynamics.training.static_lr)),
                ("discriminator", torch.optim.AdamW(discriminator_params, lr=self.config.dynamics.training.discriminator_lr)),
                (
                    "dynamic",
                    torch.optim.AdamW(
                        dynamic_params,
                        lr=self.config.dynamics.training.dynamic_lr,
                        weight_decay=self.config.dynamics.training.proj_in_weight_decay,
                    )
                ),
            ]

            if self.config.dynamics.training.iterative_dynamic_training:
                optimizers.append((
                    "dynamic_disentanglement",
                    torch.optim.AdamW(
                        self.proj_dynamic.parameters(),
                        lr=self.config.dynamics.training.dynamic_disentanglement_lr,
                        weight_decay=self.config.dynamics.training.proj_in_weight_decay,
                    ),
                ))

            if self.config.dynamics.training.update_pred:
                if self.config.dynamics.training.use_pretrained_dynamics:
                    assert self.config.dynamics.checkpoint_path is not None, "checkpoint_path must be provided if update_pred is False"
                    optimizers.append(
                        ("prediction", torch.optim.AdamW(dynamic_params, lr=self.config.dynamics.training.prediction_lr)),
                    )
                else:
                    optimizers.append(
                        ("prediction", torch.optim.AdamW(dynamic_params + other_params, lr=self.config.dynamics.training.prediction_lr)),
                    )

            # key order decides the optimization order
            optimizers = OrderedDict(optimizers)
            return optimizers

        else:
            pramas = [
                param for param in self.parameters()
                if id(param) not in encoder_params_ids
            ]
            return {"prediction": torch.optim.AdamW(pramas, lr=self.config.dynamics.training.prediction_lr)}

    def set_scheduler_step(self, step):
        self.step = step

    def load_slot_kmeans(self) -> KMeans:
        # load encodings
        data_path = Path(self.config.data.data_path)

        encodings = []
        for dataset_id in self.config.data.dataset_ids:
            encoding_path = data_path / dataset_id / "data" / self.config.data.encoding_for_kmeans_fname
            with h5py.File(encoding_path, 'r', libver="latest", swmr=True) as f:
                encodings.append(f["slots"][:])
        encodings = np.concatenate(encodings, axis=0)                                   # (num_samples, slot_dim)

        # fit kmeans
        num_clusters = self.config.dynamics.rollout.n_kmeans_clusters
        kmeans = KMeans(encodings, num_clusters=num_clusters)
        return kmeans

    def compute_dynamic_embeddings(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, float], Dict[str, torch.Tensor]]:
        assert self.disentangle_static_dynamic, "disentangle_static_dynamic must be True"

        dynamic_embeddings = self.proj_dynamic(slots)

        if self.config.dynamics.pred.dynamic_use_vq:
            dynamic_embeddings, _, commit_loss, dynamic_vq_logging = self.dynamic_vq(dynamic_embeddings)
            dynamic_vq_logging = dynamic_vq_logging.to_dict()
        else:
            commit_loss = 0.0
            dynamic_vq_logging = {}

        if self.config.dynamics.pred.normalize_dynamic_embeddings:
            dynamic_embeddings = torch.nn.functional.normalize(dynamic_embeddings, dim=-1)

        return dynamic_embeddings, commit_loss, dynamic_vq_logging

    def compute_slots_recon(self, dynamic_embeddings: torch.Tensor, static_embeddings: torch.Tensor):

        if self.static_dynamic_merge_method == "concat":
            slots_recon_input = torch.cat([dynamic_embeddings, static_embeddings], dim=-1)
        elif self.static_dynamic_merge_method == "add":
            slots_recon_input = dynamic_embeddings + static_embeddings
        elif self.static_dynamic_merge_method == "dynamic_only":
            slots_recon_input = dynamic_embeddings
        else:
            raise ValueError(f"Invalid static_dynamic_merge_method: {self.static_dynamic_merge_method}")
        slots_recon = self.slot_recon(slots_recon_input)

        return slots_recon

    def forward(
        self,
        batch: Episode,
        mode: Literal["prediction", "static", "dynamic", "discriminator", "dynamic_disentanglement"] = "prediction",
        rollout: bool=False,
        **rollout_kwargs,
    ):
        if mode != "prediction":
            assert not rollout, "rollout must be False when not in prediction mode"
            assert self.disentangle_static_dynamic, "disentangle_static_dynamic must be True"

        if mode == "discriminator":
            return self.compute_all_discriminator_loss(batch)
        elif mode == "static":
            return self.compute_static_loss(batch)
        elif mode == "dynamic":
            return self.compute_dynamic_loss(batch)
        elif mode == "dynamic_disentanglement":
            return self.compute_dynamic_disentanglement_loss(batch)
        elif mode == "prediction":

            if rollout:
                assert not self.training, "rollout must be False in training mode"
                batch, prediction = self.rollout(batch, **rollout_kwargs)
                labels, predictions = [batch], [prediction]
            else:
                if self.training:
                    assert self.config.dynamics.training.update_pred, "enter prediction training, but update_pred is False"

                # during visualization, we only need to predict one step
                seq_len = batch.seq_len
                n_pred_steps = self.config.dynamics.training.n_pred_steps if self.training else 1

                predictions = []
                labels = []

                for i in range(n_pred_steps):

                    start = i
                    end = i + seq_len - (n_pred_steps - 1)

                    input_batch = batch[start:end]
                    label = batch[start:end]

                    # randomly select 50% slots and replace them with the reconstructed slots
                    if i == 0 and self.training and self.disentangle_static_dynamic:

                        self.proj_static.eval()

                        dynamic_embeddings, commit_loss, _ = self.compute_dynamic_embeddings(input_batch.slots)
                        static_embeddings = self.proj_static(input_batch.slots).detach()
                        slots_recon = self.compute_slots_recon(dynamic_embeddings, static_embeddings)

                        bs, seq_len, num_slots, slot_dim = slots_recon.shape

                        if self.config.dynamics.training.use_pretrained_dynamics:
                            self.eval()
                            input_batch.slots = slots_recon
                        else:
                            random_mask = torch.rand(bs, seq_len, device=self.device) < 0.5
                            random_mask = repeat(random_mask, 'b t -> b t s d', s=num_slots, d=slot_dim)

                            input_batch.slots = torch.where(random_mask, slots_recon, input_batch.slots)

                    # replace input slots / enc_feat with previous step's prediction
                    if i > 0:
                        prev_prediction = predictions[-1]
                        if self.use_slot_visible_and_exists:
                            input_batch.slots_visible = prev_prediction.next_slots_visible_logits
                            input_batch.slots_exist = prev_prediction.next_slots_exist_logits

                        if self.config.dynamics.patch_as_slot:
                            input_batch.enc_feat = prev_prediction.next_slots_pred
                        else:
                            input_batch.slots = prev_prediction.next_slots_pred

                    prediction = self.compute_prediction_logits(input_batch)
                    prediction.sample(self.config.dynamics.training.deterministic)
                    predictions.append(prediction)
                    labels.append(label)

            if self.config.dynamics.training.loss_only_use_last_pred:
                loss, logging = self.compute_prediction_loss(labels[-1:], predictions[-1:])
            else:
                loss, logging = self.compute_prediction_loss(labels, predictions)

            if not rollout and self.disentangle_static_dynamic and self.config.dynamics.pred.dynamic_use_vq:
                loss = loss + commit_loss

            return labels[0], predictions[0], loss, logging

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_prediction_logits(
        self,
        input_batch: Episode,
        num_last_tokens=0,
        position_ids=None,
        inference_params=None,
        **mixer_kwargs,
    ) -> Prediction:
        """

        :param input_batch:
        :param num_last_tokens: number of steps to keep in the prediction, only used during generation
        :param position_ids: just to be compatible with Transformer generation. We don't use it.
        :param inference_params: mamba inference cache
        :param mixer_kwargs: mamba forward kwargs
        :return:
            slots: (bs, seq_len, num_slots, slot_dim)
            slots_visible, slots_exist: (bs, seq_len, num_slots)
            actions: (bs, seq_len)
        """
        if isinstance(self.config.action_space, gymnasium.spaces.Discrete):
            actions = input_batch.actions.long()                            # (bs, seq_len)
            actions_embedding = self.action_embedding(actions)              # (bs, seq_len, slot_dim)
        elif isinstance(self.config.action_space, gymnasium.spaces.Box):
            assert len(self.config.action_space.shape) == 1, "Action space must be 1D"
            actions_embedding = self.action_proj_in(input_batch.actions)    # (bs, seq_len, slot_dim)
        else:
            raise ValueError(f"Unsupported action space type: {type(self.config.action_space)}")

        if self.config.dynamics.patch_as_slot:
            slots = input_batch.enc_feat                                    # (bs, seq_len, num_slots, slot_dim)
        else:
            slots = input_batch.slots                                       # (bs, seq_len, num_slots, slot_dim)

        bs, seq_len, _, _ = slots.shape

        if self.use_slot_visible_and_exists:
            slots[~input_batch.slots_visible] = 0
        else:
            if self.training:
                assert input_batch.slots_visible[~input_batch.padding_mask].all(), "slots_visible must be True if use_slot_visible_and_exists is False"
                assert input_batch.slots_exist[~input_batch.padding_mask].all(), "slots_exist must be True if use_slot_visible_and_exists is False"

        ssm_input = slots

        if self.use_slot_visible_and_exists:
            ssm_input = torch.cat(
                [
                    ssm_input,
                    rearrange(input_batch.slots_exist, 'b t s -> b t s 1').to(self.dtype),
                    rearrange(input_batch.slots_visible, 'b t s -> b t s 1').to(self.dtype),
                ],
                dim=-1,
            )

        ssm_input = self.proj_in(ssm_input)

        # use transformer to aggregate the information across slots + the action
        ssm_input = torch.cat(
            [ssm_input, rearrange(actions_embedding, 'b t d -> b t 1 d')],
            dim=-2,
        )
        ssm_input = rearrange(ssm_input, 'b t s d -> (b t) s d')
        key_padding_mask = torch.cat(
            [rearrange(~input_batch.slots_exist, 'b t s -> (b t) s'),
             torch.zeros(bs * seq_len, 1, dtype=torch.bool, device=self.device)],
            dim=-1,
        )
        ssm_input = self.pre_ssm_mixer(ssm_input, key_padding_mask=key_padding_mask)

        ssm_input = rearrange(ssm_input, '(b t) s d -> (b s) t d', b=bs, t=seq_len)

        hidden_states = self.backbone(ssm_input, inference_params=inference_params, **mixer_kwargs)
        hidden_states = rearrange(hidden_states, '(b s) t d -> b t s d', b=bs, s=self.num_slots + 1)

        # ============ Slot Prediction ============
        # remove action token, (bs * seq_len, num_slots + 1, slot_dim) -> (bs, seq_len, num_slots, slot_dim)
        slot_pred_input = hidden_states[:, :, :-1]
        slot_pred = self.slot_predictor(slot_pred_input)

        if self.use_slot_visible_and_exists:
            (
                next_slots_delta,               # (bs, seq_len, self.num_slots, self.num_slot_modes * self.slot_dim)
                next_slots_mode_logits,         # (bs, seq_len, self.num_slots, self.num_slot_modes)
                next_slots_visible_logits,      # (bs, seq_len, num_slots, 1)
                next_slots_exist_logits,        # (bs, seq_len, num_slots, 1)
            ) = torch.split(
                slot_pred,
                [
                    self.num_slot_modes * self.slot_dim,
                    self.num_slot_modes,
                    1,
                    1,
                ],
                dim=-1
            )
            next_slots_visible_logits = next_slots_visible_logits.squeeze(-1)
            next_slots_exist_logits = next_slots_exist_logits.squeeze(-1)
        else:
            (
                next_slots_delta,               # (bs, seq_len, self.num_slots, self.num_slot_modes * self.slot_dim)
                next_slots_mode_logits,         # (bs, seq_len, self.num_slots, self.num_slot_modes)
            ) = torch.split(
                slot_pred,
                [
                    self.num_slot_modes * self.slot_dim,
                    self.num_slot_modes,
                ],
                dim=-1
            )
            bs, seq_len, num_slots, _ = next_slots_mode_logits.shape
            next_slots_visible_logits = next_slots_exist_logits = torch.ones(bs, seq_len, num_slots, dtype=torch.bool, device=self.device)

        next_slots_delta = rearrange(
            next_slots_delta,
            "b t s (m d) -> b t s m d",
            m=self.num_slot_modes,
        )

        if self.config.dynamics.pred.pred_delta:
            next_slots_dist = rearrange(slots, "b t s d -> b t s 1 d") + next_slots_delta
        else:
            next_slots_dist = next_slots_delta

        # ============ Reward and Termination Prediction ============
        pred_inputs = rearrange(hidden_states, 'b t s d -> (b t) s d')

        reward_pred_tokens = torch.cat([
            repeat(self.reward_token, '1 1 d -> bt 1 d', bt=pred_inputs.shape[0]),
            pred_inputs,
        ], dim=1)                                                               # (bs * seq_len, 1 + num_slots, d_model)
        reward_pred_tokens = self.reward_pred_mha(reward_pred_tokens)           # (bs * seq_len, 1 + num_slots, d_model)
        reward_pred_tokens = reward_pred_tokens[:, 0, :]                        # (bs * seq_len, d_model)
        rewards_logits = self.reward_pred_mlp(reward_pred_tokens)               # (bs * seq_len, num_reward_bins)
        rewards_logits = rearrange(rewards_logits, '(b t) n -> b t n', b=bs, t=seq_len)

        termination_pred_tokens = torch.cat([
            repeat(self.termination_token, '1 1 d -> bt 1 d', bt=pred_inputs.shape[0]),
            pred_inputs,
        ], dim=1)                                                               # (bs * seq_len, 1 + num_slots, d_model)
        termination_pred_tokens = self.termination_pred_mha(termination_pred_tokens)
        termination_pred_tokens = termination_pred_tokens[:, 0, :]
        terminations_logits = self.termination_pred_mlp(termination_pred_tokens)
        terminations_logits = rearrange(terminations_logits, '(b t) n -> b t n', b=bs, t=seq_len)

        prediction = Prediction(
            next_slots_dist=next_slots_dist,
            next_slots_mode_logits=next_slots_mode_logits.view(bs, seq_len, self.num_slots, self.num_slot_modes),
            next_slots_visible_logits=next_slots_visible_logits,
            next_slots_exist_logits=next_slots_exist_logits,
            rewards_logits=rewards_logits,
            terminations_logits=terminations_logits.squeeze(-1),                            # (bs, seq_len)
        )

        if num_last_tokens > 0:
            prediction_trucated = {}
            for field in fields(prediction):
                field_value = getattr(prediction, field.name)
                if isinstance(field_value, torch.Tensor):
                    prediction_trucated[field.name] = field_value[:, -num_last_tokens:]     # (bs, num_last_tokens, ...)
            prediction = dataclasses.replace(prediction, **prediction_trucated)

        return prediction

    def compute_prediction_loss(
        self,
        input_batches: List[Episode],
        predictions: List[Prediction],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # preprocess inputs
        # list of dataclasses -> dataclass
        input_batch = torch_cat_dataclasses_list(input_batches, dim=0)
        predictions = torch_cat_dataclasses_list(predictions, dim=0)

        assert input_batch.slots_exist[input_batch.slots_visible].all(), \
            "slots_exist must be True if slots_visible is True"

        current_slots_exist_in_prev_step = torch.zeros_like(input_batch.slots_exist)
        current_slots_exist_in_prev_step[:, 1:] = input_batch.slots_exist[:, :-1]

        # ============ Apply Padding Mask on (bs, seq_len,) ============

        # rewards and terminations are valid if padding_mask is False
        (
            rewards,
            terminations,
            rewards_weights,
            terminations_weights,
            rewards_logits,
            terminations_logits,
            rewards_pred,
            terminations_pred,
        ) = map(
            lambda x: None if x is None else x[~input_batch.padding_mask],
            [
                input_batch.rewards,
                input_batch.terminations,
                input_batch.rewards_weights,
                input_batch.terminations_weights,
                predictions.rewards_logits,
                predictions.terminations_logits,
                predictions.rewards_pred,
                predictions.terminations_pred,
            ],
        )

        # filter out padding in slots prediction
        padding_mask = input_batch.padding_mask[:, 1:]      # (bs, seq_len - 1)
        (
            next_slots_label,                               # (SUM(seq_len in batch), num_slots, slot_dim or slot_dim)
            next_slots_visible_label,                       # (SUM(seq_len in batch), num_slots)
            next_slots_exist_label,                         # (SUM(seq_len in batch), num_slots)
            next_slots_exist_in_current_step,               # (SUM(seq_len in batch), num_slots)
        ) = map(
            lambda x: x[:, 1:][~padding_mask],              # the first step won't be predicted
            [
                input_batch.enc_feat if self.config.dynamics.patch_as_slot else input_batch.slots,
                input_batch.slots_visible,
                input_batch.slots_exist,
                current_slots_exist_in_prev_step,
            ],
        )

        (
            next_slots_dist,                        # (SUM(seq_len in batch), num_slots, num_slot_modes, slot_dim or slot_dim)
            next_slots_mode_logits,                 # (SUM(seq_len in batch), num_slots, num_slot_modes)
            next_slots_visible_logits,              # (SUM(seq_len in batch), num_slots)
            next_slots_exist_logits,                # (SUM(seq_len in batch), num_slots)
            next_slots_visible_pred,                # (SUM(seq_len in batch), num_slots)
            next_slots_exist_pred,                  # (SUM(seq_len in batch), num_slots)
        ) = map(
            lambda x: None if x is None else x[:, :-1][~padding_mask],  # the last step has no label
            [
                predictions.next_slots_dist,
                predictions.next_slots_mode_logits,
                predictions.next_slots_visible_logits,
                predictions.next_slots_exist_logits,
                predictions.next_slots_visible_pred,
                predictions.next_slots_exist_pred,
            ],
        )

        # tp1: t + 1
        exist_t_exist_tp1 = next_slots_exist_in_current_step & next_slots_exist_label
        exist_t_not_exist_tp1 = next_slots_exist_in_current_step & ~next_slots_exist_label
        not_exist_t_exist_tp1 = ~next_slots_exist_in_current_step & next_slots_exist_label
        not_exist_t_not_exist_tp1 = ~next_slots_exist_in_current_step & ~next_slots_exist_label
        assert next_slots_visible_label[not_exist_t_exist_tp1].all(), "just appeared slots must be visible"
        assert not next_slots_visible_label[exist_t_not_exist_tp1].any(), "non-existing slots must be invisible"
        assert not next_slots_visible_label[not_exist_t_not_exist_tp1].any(), "non-existing slots must be invisible"

        # ============ Slot Value Prediction ============
        if self.use_slot_visible_and_exists:
            if self.config.dynamics.pred.slot_pred.use_assignment:
                # ------ for each next_slots_label, use dynamic assignment to assign a next_slots prediction ------
                # assigned_pred_idxes: (SUM(seq_len in batch), num_slots),
                #   the index of the assigned prediction for each slot label
                # pred_assigned: (SUM(seq_len in batch), num_slots),
                #   whether the prediction is assigned to a valid label
                # slot_disappear_due_to_sam_error: (SUM(seq_len in batch), num_slots),
                #   whether the slot disappears due to sam error
                assigned_pred_idxes, pred_assigned, slot_disappear_due_to_sam_error = self.assign_pred_to_label(
                    next_slots_label, next_slots_dist,
                    exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_not_exist_tp1,
                    next_slots_exist_label, next_slots_exist_in_current_step,
                )

                # rearrange next slots predictions according to assigned_pred_idxes
                # (SUM(seq_len in batch), num_slots, num_slot_modes, slot_dim)
                next_slots_dist = next_slots_dist.gather(
                    1,
                    repeat(assigned_pred_idxes, 'b s -> b s m d', m=self.num_slot_modes, d=self.slot_dim),
                )

                # (SUM(seq_len in batch), num_slots, num_slot_modes)
                next_slots_mode_logits = next_slots_mode_logits.gather(
                    1,
                    repeat(assigned_pred_idxes, 'b s -> b s m', m=self.num_slot_modes),
                )

                optimize_cases = {"case_I": exist_t_exist_tp1 & next_slots_exist_label}
                if self.config.dynamics.pred.pred_appearing_slots:
                    optimize_cases["case_III"] = not_exist_t_exist_tp1 & next_slots_exist_label
            else:
                optimize_cases = {
                    "case_I": exist_t_exist_tp1,
                    "case_II": exist_t_not_exist_tp1,
                    "case_III": not_exist_t_exist_tp1,
                    "case_IV": not_exist_t_not_exist_tp1,
                }
        else:
            optimize_cases = {
                "case_I": torch.ones_like(exist_t_exist_tp1, dtype=torch.bool),
            }

        optimize_mask = torch.sum(torch.stack(list(optimize_cases.values())), dim=0).bool()

        # (SUM(seq_len in batch), num_slots, slot_dim) -> (SUM(seq_len in batch), num_slots, num_slot_modes, slot_dim)
        next_slots_label_expan = repeat(next_slots_label, 'b s d -> b s m d', m=self.num_slot_modes)

        # (SUM(seq_len in batch), num_slots, num_slot_modes)
        slot_pred_error = self.slot_loss_fn(next_slots_dist, next_slots_label_expan.detach(), reduction="none").mean(dim=-1)

        # when using multi-modal prediction, only optimize the mode with the smallest prediction error
        # (SUM(seq_len in batch), num_slots), (SUM(seq_len in batch), num_slots)
        slot_pred_min_error, min_error_idx = slot_pred_error.min(dim=-1)

        # (SUM(seq_len in batch), num_slots, 1, slot_dim)
        selected_next_slots_pred = next_slots_dist.gather(
            -2,
            # (SUM(seq_len in batch), num_slots) -> (SUM(seq_len in batch), num_slots, slot_dim)
            min_error_idx[:, :, None, None].expand(-1, -1, 1, self.slot_dim),
        )
        selected_next_slots_pred = rearrange(selected_next_slots_pred, 'b s 1 d -> b s d')

        slots_loss = slot_pred_min_error[optimize_mask].mean()

        slot_abs_error = F.l1_loss(next_slots_dist, next_slots_label_expan.detach(), reduction="none").mean(dim=-1)
        slot_abs_error, _ = slot_abs_error.min(dim=-1)

        logging = {
            "slots_loss": slots_loss,
            "slots_relative_error": slot_abs_error[optimize_mask].mean() / next_slots_label[optimize_mask].std(dim=0).mean(),
        }

        for case, mask in optimize_cases.items():
            if mask.int().sum() < 2:
                continue
            slot_loss_case = slot_abs_error[mask].mean()
            logging.update({
                f"slots_loss_{case}": slot_loss_case,
                f"slots_relative_error_{case}": slot_loss_case / next_slots_label[mask].std(dim=0).mean(),
            })

        if self.num_slot_modes > 1:
            slots_mode_loss = F.cross_entropy(next_slots_mode_logits[optimize_mask], min_error_idx[optimize_mask])
            slots_mode_accuracy = (min_error_idx == next_slots_mode_logits.argmax(dim=-1)).float().mean()

            logging.update({
                "slots_mode_loss": slots_mode_loss,
                "slots_mode_accuracy": slots_mode_accuracy,
            })

        # ============ Enc Feat, Obs Prediction ============
        if self.config.dynamics.training.pred_enc_feat or self.config.dynamics.training.pred_obs:
            assert not self.config.encoder.use_sam_mask
            assert self.config.dynamics.training.pred_ratio > 0

            # only decode a subset of the slots to save memory
            bs = selected_next_slots_pred.shape[0]
            random_mask = torch.zeros(bs, device=self.config.device, dtype=torch.bool)
            random_mask[:max(1, int(bs * self.config.dynamics.training.pred_ratio))] = True
            random_mask = random_mask[torch.randperm(bs)]

            if torch.any(random_mask):

                self.encoder.eval()
                if self.config.dynamics.patch_as_slot:
                    if self.config.dynamics.training.pred_obs:
                        next_obs_pred = self.encoder.decode_enc_feat_to_rgb(selected_next_slots_pred[random_mask], requires_grad=True)
                        next_obs_label = input_batch.observations[:, 1:][~padding_mask]

                        logging["obs_loss"] = F.mse_loss(next_obs_pred, next_obs_label[random_mask])
                else:
                    modes = []
                    if self.config.dynamics.training.pred_enc_feat:
                        modes.append("enc_feat_rec")
                    if self.config.dynamics.training.pred_obs:
                        modes.append("rgb_rec")

                    outputs = self.encoder.decode(selected_next_slots_pred[random_mask], modes=modes, requires_grad=True)

                    # list -> dict
                    if len(modes) == 1:
                        outputs = [outputs]
                    outputs = {k: v for k, v in zip(modes, outputs)}

                    if self.config.dynamics.training.pred_enc_feat:
                        next_enc_feat_pred = outputs["enc_feat_rec"]
                        next_enc_feat_label = input_batch.enc_feat[:, 1:][~padding_mask]

                        logging["enc_feat_loss"] = F.mse_loss(next_enc_feat_pred, next_enc_feat_label[random_mask])

                    if self.config.dynamics.training.pred_obs:
                        next_obs_pred = outputs["rgb_rec"]
                        next_obs_label = input_batch.observations[:, 1:][~padding_mask]

                        logging["obs_loss"] = F.mse_loss(next_obs_pred, next_obs_label[random_mask])

        # ============ Slot Visible, Existence Prediction ============
        # If a slot is assigned to a valid label, we optimize the prediction
        # If a slot is not assigned to any valid label, it should be invisible and non-existent (i.e., label = False)

        if self.use_slot_visible_and_exists:
            if self.config.dynamics.pred.slot_pred.use_assignment:
                get_pred_and_label = partial(
                    self.get_pred_and_label_with_assignment,
                    exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_exist_tp1,
                    assigned_pred_idxes, pred_assigned, slot_disappear_due_to_sam_error,
                )
            else:
                get_pred_and_label = partial(
                    self.get_pred_and_label_without_assignment,
                    exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_exist_tp1, not_exist_t_not_exist_tp1,
                )
            next_slots_visible_logits, next_slots_visible_label_ = get_pred_and_label(
                next_slots_visible_logits, next_slots_visible_label
            )
            next_slots_exist_logits, next_slots_exist_label_ = get_pred_and_label(
                next_slots_exist_logits, next_slots_exist_label
            )

            slots_visible_loss = F.binary_cross_entropy_with_logits(
                torch.cat([v for v in next_slots_visible_logits.values()]),
                torch.cat([v for v in next_slots_visible_label_.values()]),
            )
            slots_exist_loss = F.binary_cross_entropy_with_logits(
                torch.cat([v for v in next_slots_exist_logits.values()]),
                torch.cat([v for v in next_slots_exist_label_.values()]),
            )
            logging.update({
                "slots_visible_loss": slots_visible_loss,
                "slots_exist_loss": slots_exist_loss,
            })

            if next_slots_visible_pred is None:
                next_slots_visible_pred = {k: v > 0 for k, v in next_slots_visible_logits.items()}
            else:
                next_slots_visible_pred = get_pred_and_label(next_slots_visible_pred)

            if next_slots_exist_pred is None:
                next_slots_exist_pred = {k: v > 0 for k, v in next_slots_exist_logits.items()}
            else:
                next_slots_exist_pred = get_pred_and_label(next_slots_exist_pred)

            for pred, label, tag in zip(
                [next_slots_visible_pred, next_slots_exist_pred],
                [next_slots_visible_label_, next_slots_exist_label_],
                ["visible", "exist"],
            ):
                for case_name in pred:
                    pred_case, label_case = pred[case_name], label[case_name]
                    if len(pred_case) == 0:
                        continue
                    accuracy = (pred_case == label_case.long()).float().mean()
                    logging[f"slots_{tag}_{case_name}_accuracy"] = accuracy

        # ============ Reward Prediction ============
        rewards_two_hot = two_hot(symlog(rewards), num_buckets=self.config.dynamics.pred.num_reward_bins)
        rewards_loss = F.cross_entropy(rewards_logits, rewards_two_hot, reduction="none")

        if self.config.dynamics.training.reward_focal_loss_gamma > 0:
            raise NotImplementedError("Focal loss for reward prediction is not implemented")

        if self.config.dynamics.training.use_reward_termination_weight:
            rewards_loss = (rewards_loss * rewards_weights).mean()
        else:
            rewards_loss = rewards_loss.mean()

        logging["rewards_loss"] = rewards_loss

        if rewards_pred is None:
            rewards_pred = symexp(compute_softmax_over_buckets(rewards_logits))

        for tag in ["negative", "zero", "positive"]:
            if tag == "negative":
                mask = rewards < 0.
            elif tag == "zero":
                mask = rewards == 0.
            else:
                mask = rewards > 0
            if not mask.any():
                continue
            error = (rewards_pred - rewards)[mask].abs().mean()
            logging[f"rewards_{tag}_error"] = error

        # ============ Done Prediction ============
        terminations_loss = F.binary_cross_entropy_with_logits(terminations_logits, terminations.float(), reduction="none")

        if self.config.dynamics.training.termination_focal_loss_gamma > 0:
            log_prob = -terminations_loss
            prob = torch.exp(log_prob)
            terminations_loss = (1 - prob) ** self.config.dynamics.training.termination_focal_loss_gamma * terminations_loss

        if self.config.dynamics.training.use_reward_termination_weight:
            terminations_loss = (terminations_loss * terminations_weights).mean()
        else:
            terminations_loss = terminations_loss.mean()

        logging["terminations_loss"] = terminations_loss

        if terminations_pred is None:
            terminations_pred = terminations_logits > 0

        for val, val_tag in zip([1., 0.], ["true", "false"]):
            mask = terminations == val
            if not mask.any():
                continue
            accuracy = (terminations_pred == terminations.long())[mask].float().mean()
            logging[f"terminations_{val_tag}_accuracy"] = accuracy

        loss = sum([v for k, v in logging.items() if k.endswith("_loss")])

        return loss, logging

    def compute_static_loss(
        self,
        input_batch: Episode,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # in this function, b: batch size, t: seq_len, s: num_slots, d: slot_dim, n: num_visible_slots

        static_embeddings = self.proj_static(input_batch.slots)

        shifted_static_embeddings = static_embeddings.gather(
            dim=1,
            index=repeat(input_batch.temporal_random_idxes, 'b t s -> b t s d', d=self.slot_dim),
        )
        static_embeddings, shifted_static_embeddings = map(
            lambda x: rearrange(x, 'b t s d -> b s t d'),
            [
                static_embeddings,
                shifted_static_embeddings,
            ],
        )

        slots_visible, slot_obj_id = map(
            lambda x: rearrange(x, 'b t s -> b s t'),
            [input_batch.slots_visible, input_batch.slot_obj_id],
        )

        # a slot can be occupied by multiple objects at different timestamps, and if the object pair at t and t + n
        # are not the same, we should not consider them as a positive pair for contrastive loss
        seq_len = static_embeddings.shape[-2]
        positive_static_embeddings = repeat(static_embeddings, 'b s t d -> b s t1 t d', t1=seq_len)
        positive_valid = slot_obj_id.unsqueeze(-1) == slot_obj_id.unsqueeze(-2)

        # (bs, seq_len, num_slots, ...) -> (num_visible_slots, ...)
        (
            static_embeddings,
            positive_static_embeddings,
            positive_valid,
            shifted_static_embeddings,
        ) = map(
            lambda x: x[slots_visible],
            [
                static_embeddings,
                positive_static_embeddings,
                positive_valid,
                shifted_static_embeddings,
            ],
        )

        logging = {}

        # =========== Static @ t vs Static @ t' Contrastive Loss ===========
        # positive pair: static @ t and static @ t' from the same obj
        # negative pair: static @ t and static from different slots

        normalize_static_embeddings = F.normalize(static_embeddings, dim=-1)
        normalize_shifted_static_embeddings = F.normalize(shifted_static_embeddings, dim=-1)

        # (num_visible_slots, seq_len)
        positive_score = torch.einsum("nd, nd -> n", normalize_static_embeddings, normalize_shifted_static_embeddings)
        logging["static_positive_cosine_similarity"] = positive_score.mean()

        # (num_visible_slots, num_negatives)
        num_negatives = self.config.dynamics.training.contrastive_num_negatives
        negative_score = (normalize_static_embeddings[:, None] * normalize_static_embeddings[None, :num_negatives]).sum(dim=-1)

        # for each i, j, if i - j < seq_len or j - i < seq_len, then i, j may belong to the same object and thus not a negative pair
        disentangle_seq_len = self.config.dynamics.training.disentangle_seq_len
        full = torch.ones_like(negative_score, dtype=torch.bool)
        mask = torch.triu(full, diagonal=disentangle_seq_len) | torch.tril(full, diagonal=-disentangle_seq_len)
        negative_score.masked_fill_(~mask, float("-inf"))
        logging["static_negative_cosine_similarity"] = negative_score[mask].mean()

        positive_score = positive_score / self.config.dynamics.training.contrastive_temperature
        negative_score = negative_score / self.config.dynamics.training.contrastive_temperature
        contrastive_loss = -(positive_score - torch.logsumexp(negative_score, dim=1)).mean()

        logging["static_contrastive_loss"] = contrastive_loss * self.config.dynamics.training.static_contrastive_coef

        # =========== Static @ t vs Static @ t' Difference Loss ===========
        static_embeddings_expan = repeat(static_embeddings, 'n d -> n t d', t=seq_len)
        valid_static_embeddings = static_embeddings_expan[positive_valid]
        valid_positive_static_embeddings = positive_static_embeddings[positive_valid]

        diff_loss = self.slot_loss_fn(valid_static_embeddings, valid_positive_static_embeddings)

        logging["static_norm"] = valid_static_embeddings.norm(dim=-1).mean()
        logging["static_difference"] = diff_loss
        logging["static_difference_loss"] = diff_loss * self.config.dynamics.training.static_invariance_coef

        static_loss = sum([v for k, v in logging.items() if k.endswith("_loss")])

        return static_loss, logging

    def compute_dynamic_loss(
        self,
        input_batch: Episode,
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        # in this function, b: batch size, t: seq_len, s: num_slots, d: slot_dim, n: num_visible_slots

        slots = input_batch.slots                                                                       # (b, t, s, d)
        slots_visible = input_batch.slots_visible                                                       # (b, t, s)
        padding_mask = input_batch.padding_mask                                                         # (b, t)
        assert not slots_visible[padding_mask].any(), "slots_visible should be False at padded positions"

        self.proj_static.eval()
        static_embeddings = self.proj_static(slots).detach()                                            # (b, t, s, d)

        dynamic_embeddings, commit_loss, dynamic_vq_logging = self.compute_dynamic_embeddings(slots)

        logging = {
            "commitment_loss": commit_loss,
            "dynamic_norm": dynamic_embeddings.norm(dim=-1).mean(),
        }
        logging.update({f"codebook_{k}": v for k, v in dynamic_vq_logging.items()})

        # ============ Slot, Enc Feat, Obs Reconstruction ============

        dynamic_embeddings_dropout = F.dropout(dynamic_embeddings, p=self.config.dynamics.training.dynamic_recon_dropout)

        slots_recon = self.compute_slots_recon(dynamic_embeddings_dropout, static_embeddings)

        visible_slots = slots[slots_visible]
        visible_slots_recon = slots_recon[slots_visible]

        slots_recon_loss = self.slot_loss_fn(visible_slots, visible_slots_recon, reduction="mean")
        logging["slots_recon_loss"] = slots_recon_loss
        logging["slots_recon_relative_error"] = F.l1_loss(visible_slots, visible_slots_recon) / visible_slots.std(dim=0).mean()

        if self.config.dynamics.training.pred_enc_feat or self.config.dynamics.training.pred_obs:
            assert not self.config.encoder.use_sam_mask
            assert self.config.dynamics.training.pred_ratio > 0

            slots_recon = slots_recon[~padding_mask]                                                    # (num_frames, s, d) 

            # only decode a subset of the slots to save memory
            num_frames = slots_recon.shape[0]
            random_mask = torch.zeros(num_frames, device=self.config.device, dtype=torch.bool)
            random_mask[:max(1, int(num_frames * self.config.dynamics.training.pred_ratio))] = True
            random_mask = random_mask[torch.randperm(num_frames)]

            selected_slots_recon = slots_recon[random_mask]

            if torch.any(random_mask):
                modes = []
                if self.config.dynamics.training.pred_enc_feat:
                    modes.append("enc_feat_rec")
                if self.config.dynamics.training.pred_obs:
                    modes.append("rgb_rec")

                self.encoder.eval()
                outputs = self.encoder.decode(selected_slots_recon, modes=modes, requires_grad=True)

                # list -> dict
                if len(modes) == 1:
                    outputs = [outputs]
                outputs = {k: v for k, v in zip(modes, outputs)}

                if self.config.dynamics.training.pred_enc_feat:
                    enc_feat_recon = outputs["enc_feat_rec"]
                    enc_feat = input_batch.enc_feat[~padding_mask]

                    logging["enc_feat_recon_loss"] = F.mse_loss(enc_feat_recon, enc_feat[random_mask])

                if self.config.dynamics.training.pred_obs:
                    obs_recon = outputs["rgb_rec"]
                    obs = input_batch.observations[~padding_mask]

                    logging["obs_recon_loss"] = F.mse_loss(obs_recon, obs[random_mask])

        # ================= Dynamic @ t vs Static @ t Disentanglement =================

        shifted_dynamic_embeddings = dynamic_embeddings.gather(
            dim=1,
            index=repeat(input_batch.temporal_random_idxes, 'b t s -> b t s d', d=self.slot_dim),
        )

        # (bs, seq_len, num_slots, ...) -> (num_visible_slots, ...)
        (
            dynamic_embeddings,
            static_embeddings,
            shifted_dynamic_embeddings,
        ) = map(
            lambda x: x[slots_visible],
            [
                dynamic_embeddings,
                static_embeddings,
                shifted_dynamic_embeddings,
            ],
        )

        if not self.config.dynamics.training.iterative_dynamic_training:
            _, disentanglement_logging = self.compute_dynamic_disentanglement_loss(
                static_embeddings=static_embeddings,
                dynamic_embeddings=dynamic_embeddings,
                shifted_dynamic_embeddings=shifted_dynamic_embeddings,
            )
            logging.update(disentanglement_logging)

        dynamic_losses = {k: v for k, v in logging.items() if k.endswith("_loss")}

        if self.config.dynamics.training.use_ca_grad and self.config.dynamics.training.dynamic_adversarial_loss_weight_schedule.get_value(self.step) > 0:
            dynamic_adversarial_loss = dynamic_losses.pop("dynamic_adversarial_loss")
            dynamic_other_loss = sum(dynamic_losses.values())
            dynamic_loss = {
                "adversarial": dynamic_adversarial_loss,
                "other": dynamic_other_loss,
            }
        else:
            dynamic_loss = sum(dynamic_losses.values())

        return dynamic_loss, logging

    def compute_dynamic_disentanglement_loss(
        self,
        input_batch: Optional[Episode] = None,
        static_embeddings: Optional[torch.Tensor] = None,
        dynamic_embeddings: Optional[torch.Tensor] = None,
        shifted_dynamic_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        # ================= Dynamic @ t vs Static @ t Disentanglement =================
        # in this function, b: batch size, t: seq_len, s: num_slots, d: slot_dim, n: num_visible_slots

        if static_embeddings is None and dynamic_embeddings is None:
            assert input_batch is not None

            slots = input_batch.slots                                                                       # (b, t, s, d)
            slots_visible = input_batch.slots_visible                                                       # (b, t, s)
            padding_mask = input_batch.padding_mask                                                         # (b, t)
            assert not slots_visible[padding_mask].any(), "slots_visible should be False at padded positions"

            self.proj_static.eval()
            static_embeddings = self.proj_static(slots).detach()                                            # (b, t, s, d)
            dynamic_embeddings, _, _ = self.compute_dynamic_embeddings(slots)                               # (b, t, s, d)

            shifted_dynamic_embeddings = dynamic_embeddings.gather(
                dim=1,
                index=repeat(input_batch.temporal_random_idxes, 'b t s -> b t s d', d=self.slot_dim),
            )

            # (bs, seq_len, num_slots, ...) -> (num_visible_slots, ...)
            (
                dynamic_embeddings,
                static_embeddings,
                shifted_dynamic_embeddings,
            ) = map(
                lambda x: x[slots_visible],
                [
                    dynamic_embeddings,
                    static_embeddings,
                    shifted_dynamic_embeddings,
                ],
            )
        else:
            assert static_embeddings is not None
            assert dynamic_embeddings is not None
            assert shifted_dynamic_embeddings is not None

        normalized_static_embeddings = F.normalize(static_embeddings, dim=-1)
        normalize_dynamic_embeddings = F.normalize(dynamic_embeddings, dim=-1)
        normalize_shifted_dynamic_embeddings = F.normalize(shifted_dynamic_embeddings, dim=-1)

        logging = {}

        # (num_visible_slots, 2 * slot_dim)
        positive_dynamic_static_concat = torch.cat([normalize_dynamic_embeddings, normalized_static_embeddings], dim=-1)

        self.dynamic_to_static_discriminator.eval()
        positive_logits = self.dynamic_to_static_discriminator(positive_dynamic_static_concat)[..., 0]
        positive_logits = torch.where(positive_logits > 0, positive_logits, torch.zeros_like(positive_logits))

        if self.config.dynamics.training.discriminator_type == "logistic":
            score = torch.mean(F.softplus(positive_logits))
        elif self.config.dynamics.training.discriminator_type == "wasserstein":
            score = torch.mean(positive_logits)
        else:
            raise ValueError(f"Unknown discriminator type: {self.config.dynamics.training.discriminator_type}")

        anneal_coef = self.config.dynamics.training.dynamic_adversarial_loss_weight_schedule.get_value(self.step)

        logging["dynamic_adversarial_score"] = score
        logging["dynamic_adversarial_loss"] = anneal_coef * score

        # ================= Dynamic @ t vs Dynamic @ t' Constrastive Loss =================
        # this avoids mode collapse

        normalize_dynamic_embeddings = F.normalize(dynamic_embeddings, dim=-1)

        # (num_visible_slots, num_visible_slots)
        score = torch.einsum("nd, nd -> n", normalize_dynamic_embeddings, normalize_shifted_dynamic_embeddings)
        logging["dynamic_cosine_similarity"] = score.mean()

        overfit = F.relu(score - self.config.dynamics.training.dynamic_contrastive_threshold).mean()
        overfit = overfit / self.config.dynamics.training.contrastive_temperature
        logging["dynamic_contrastive_loss"] = overfit * self.config.dynamics.training.dynamic_contrastive_coef


        dynamic_disentanglement_loss = sum([v for k, v in logging.items() if k.endswith("_loss")])

        return dynamic_disentanglement_loss, logging

    def compute_all_discriminator_loss(
        self,
        input_batch: Episode,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            self.proj_dynamic.eval()
            self.proj_static.eval()

            dynamic_embeddings = self.proj_dynamic(input_batch.slots)
            static_embeddings = self.proj_static(input_batch.slots)

            static_embeddings = F.normalize(static_embeddings, dim=-1)
            dynamic_embeddings = F.normalize(dynamic_embeddings, dim=-1)

            # (bs, seq_len, num_slots, ...) -> (num_visible_slots, ...)
            dynamic_embeddings, static_embeddings = map(
                lambda x: x[input_batch.slots_visible],
                [dynamic_embeddings, static_embeddings],
            )
            num_visible_slots = dynamic_embeddings.shape[0]

            shifted_static_embeddings = static_embeddings.roll(dims=0, shifts=num_visible_slots // 2)
            dynamic_positive_input = torch.cat([dynamic_embeddings, static_embeddings], dim=-1)
            dynamic_negative_input = torch.cat([dynamic_embeddings, shifted_static_embeddings], dim=-1)

        # ================= Dynamic @ t vs Static @ t Disentanglement =================
        dynamic_discriminator_logging = self.compute_discriminator_loss(
            self.dynamic_to_static_discriminator,
            dynamic_positive_input,
            dynamic_negative_input,
            "dynamic",
        )

        discriminator_loss = sum([v for k, v in dynamic_discriminator_logging.items() if k.endswith("_loss")])

        return discriminator_loss, dynamic_discriminator_logging

    def compute_discriminator_loss(
        self,
        discriminator,
        positive_input,
        negative_input,
        log_prefix="",
    ) -> Dict[str, torch.Tensor]:
        assert positive_input.shape == negative_input.shape
        assert positive_input.ndim == 2

        # ============ Discrimination Loss ============
        all_input = torch.cat([positive_input, negative_input], dim=0)
        all_logits = discriminator(all_input)[..., 0]
        positive_logits, negative_logits = torch.chunk(all_logits, 2, dim=0)

        logging = {}
        if self.config.dynamics.training.discriminator_type == "logistic":
            loss = torch.mean(F.softplus(-positive_logits)) + torch.mean(F.softplus(negative_logits))
            logging[f"{log_prefix}_discriminator_accuracy"] = (
                (positive_logits >= 0).float().mean() + (negative_logits < 0).float().mean()
            ) / 2
        elif self.config.dynamics.training.discriminator_type == "wasserstein":
            loss = torch.mean(F.relu(1. - positive_logits) + F.relu(1. + negative_logits))
            logging[f"{log_prefix}_discriminator_accuracy"] = (positive_logits > negative_logits).float().mean()
        else:
            raise ValueError(f"Unknown discriminator type: {self.config.dynamics.training.discriminator_type}")

        logging[f"{log_prefix}_discriminator_loss"] = loss

        if self.discriminator_regularization_type == "gradient_penalty":
            # ============ Gradient Penalty ============
            alpha = torch.rand(positive_input.shape[0], 1, **self.factory_kwargs)

            interpolates = alpha * positive_input + (1 - alpha) * negative_input
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
            outputs = discriminator(interpolates)
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=interpolates,
                grad_outputs=torch.ones(outputs.size(), **self.factory_kwargs),
                create_graph=True,
                retain_graph=True,
            )[0]
            grads = grads.view(grads.size(0), -1)

            gradient_penalty_coef = self.config.dynamics.training.discriminator_gradient_penalty_coef
            grad_penalty = (grads.norm(2, dim=1) - 1).pow(2).mean()

            logging[f"{log_prefix}_discriminator_grad_penalty"] = grad_penalty
            logging[f"{log_prefix}_discriminator_grad_penalty_loss"] = grad_penalty * gradient_penalty_coef
        elif self.discriminator_regularization_type == "lecam":
            self.lecam_ema.update(positive_logits, negative_logits)
            lecam_reg = self.lecam_ema.lecam_reg(positive_logits, negative_logits)

            lecam_coef = self.config.dynamics.training.discriminator_lecam_coef
            logging[f"{log_prefix}_discriminator_lecam_reg"] = lecam_reg
            logging[f"{log_prefix}_discriminator_lecam_reg_loss"] = lecam_reg * lecam_coef
        else:
            raise ValueError(f"Unknown discriminator regularization type: {self.discriminator_regularization_type}")

        return logging

    def assign_pred_to_label(
        self,
        next_slots_label, next_slots_dist,
        exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_not_exist_tp1,
        next_slots_exist_label, next_slots_exist_in_current_step,
    ):
        # for each next_slots_label, there are four cases:
        # I. exist @ t and exist @ t + 1:
        #   use the same slot as the prediction
        #   optimize next_slots, next_slots_visible_logits and next_slots_exist_logits
        # II. exist @ t but not exist @ t + 1:
        #   1. if natural and predictable:
        #       use the same slot as the prediction
        #       optimize next_slots_visible_logits and next_slots_exist_logits
        #   2. if caused by sam error (i.e., if its prediction is assigned to another just appeared slots label),
        #       do not optimize any prediction
        # III. not exist @ t but exist @ t + 1:
        #   use the slot with the minimal prediction error
        #   1. if natural and predictable (i.e., agent shoots a bullet),
        #       optimize next_slots, next_slots_visible_logits and next_slots_exist_logits
        #   2. if natural but unpredictable (i.e., object showing up from boundary),
        #       do not optimize any prediction
        #   3. if caused by sam error (i.e., the label is assigned to an "exist @ t but not exist @ t + 1" slot),
        #       optimize next_slots, next_slots_visible_logits and next_slots_exist_logits
        # IV. neither exist @ t nor exist @ t + 1: = ~next_slots_exist_in_current_step & ~next_slots_exist_label
        #   do not assign any slot nor optimize any prediction
        with torch.inference_mode():
            # distance matrix (SUM(seq_len in batch), num_slots, num_slots * num_slot_modes)
            cost_matrix = torch.cdist(next_slots_label, next_slots_dist.flatten(1, 2), p=1) / self.slot_dim

            # take the min error across prediction modes (SUM(seq_len in batch), num_slots, num_slots)
            # cost_matrix[b, i, j]: the prediction error of assigning next_slots_label[b, i] to next_slots[b, j]
            cost_matrix = cost_matrix.view(-1, self.num_slots, self.num_slots, self.num_slot_modes).min(dim=-1).values

            # Case I: exist @ t and exist @ t + 1
            off_diagonal = ~torch.eye(self.num_slots, dtype=torch.bool, device=self.device).expand_as(cost_matrix)
            exist_t_exist_tp1_assignment_mask = torch.zeros_like(cost_matrix, dtype=torch.bool)
            exist_t_exist_tp1_expan = exist_t_exist_tp1.unsqueeze(-1).expand_as(cost_matrix)

            # if exist_t_exist_tp1[b, i] = True, exist_t_exist_tp1_assignment_mask[b, i, j != i] = True
            exist_t_exist_tp1_assignment_mask[exist_t_exist_tp1_expan] = off_diagonal[exist_t_exist_tp1_expan]

            # if exist_t_exist_tp1[b, i] = True,
            # exist_t_exist_tp1_assignment_mask[b, i, j != i] = True,
            # exist_t_exist_tp1_assignment_mask[b, j != i, i] = True
            exist_t_exist_tp1_assignment_mask = \
                exist_t_exist_tp1_assignment_mask | exist_t_exist_tp1_assignment_mask.transpose(1, 2)

            # cost_matrix[exist_t_exist_tp1_assignment_mask] = float("inf")

            # Case II: exist @ t but not exist @ t + 1
            # no slot should be assigned during dynamic assignment, we will overwrite the assignment later
            cost_matrix[exist_t_not_exist_tp1] = float("inf")

            # Case IV: neither exist @ t nor exist @ t + 1,
            # no slot should be assigned, so we set cost_matrix[b, i, :] = inf
            cost_matrix[not_exist_t_not_exist_tp1] = float("inf")

            soft_assignment_method = self.config.dynamics.pred.slot_pred.soft_assignment_method
            hard_assignment_method = self.config.dynamics.pred.slot_pred.hard_assignment_method

            if soft_assignment_method == "vanilla":
                pass
            elif soft_assignment_method == "sinkhorn":
                # using sinkhorn algorithm to refine the assignment
                # log_sinkhorn_algorithm returns log_prob, so we need to add "-" to convert it to the cost
                sinkhorn_cfg = dataclasses.asdict(self.config.dynamics.pred.slot_pred.sinkhorn_cfg)
                cost_matrix = -log_sinkhorn_algorithm(
                    cost_matrix,
                    **sinkhorn_cfg,
                )
            else:
                raise ValueError(f"Invalid soft_assignment_method: {soft_assignment_method}")

            # assigned_pred_idxes: (SUM(seq_len in batch), num_slots)
            # label at [b, j] is assigned to a prediction at assigned_pred_idxes[b, i]
            if hard_assignment_method == "greedy":
                assigned_pred_idxes = cost_matrix.argmin(dim=-1)
            else:
                raise ValueError(f"Invalid hard_assignment_method: {hard_assignment_method}")

            assigned_pred_idxes[not_exist_t_not_exist_tp1] = -1

            exist_tp1_label_assigned_pred_idxes = assigned_pred_idxes.clone()

            # overwrite the assignment for Case II: exist @ t but not exist @ t + 1
            identity_idxes = torch.arange(self.num_slots, device=self.device).expand_as(assigned_pred_idxes)
            assigned_pred_idxes[exist_t_not_exist_tp1] = identity_idxes[exist_t_not_exist_tp1]

            # assert torch.all(assigned_pred_idxes[next_slots_exist_in_current_step] != -1), \
            #     "valid slots label must be assigned with a prediction"
            assert torch.all(assigned_pred_idxes[next_slots_exist_label] != -1), \
                "valid slots label must be assigned with a prediction"
            # assert torch.all((assigned_pred_idxes == identity_idxes)[next_slots_exist_in_current_step]), \
            #     "existing slots must be assigned to themselves"

            # assign invalid labels to the first slot, otherwise torch.gather will have index out of bounds error
            assigned_pred_idxes[assigned_pred_idxes == -1] = 0

            sum_seq_len = next_slots_label.shape[0]

            # get pred_assigned, a binary mask of shape (SUM(seq_len in batch), num_slots):
            #   for whether the slot prediction at [b, i] is assigned to a label
            # this helps us identify the slots that are not assigned to any label
            pred_assigned = torch.zeros(sum_seq_len, self.num_slots + 1, dtype=torch.bool, device=self.device)
            pred_assigned[torch.arange(sum_seq_len).unsqueeze(1), assigned_pred_idxes] = assigned_pred_idxes != -1
            pred_assigned = pred_assigned[:, :-1]

            # get slots_assignment_valid, a binary mask of shape (SUM(seq_len in batch), num_slots):
            #   for whether the slot prediction at [b, i] is assigned to a label where exist @ t + 1
            # this helps us to filter out slot disappearances caused by sam error
            pred_assigned_to_exist_tp1_label = torch.zeros(
                sum_seq_len, self.num_slots + 1,
                dtype=torch.bool, device=self.device,
            )
            pred_assigned_to_exist_tp1_label[
                torch.arange(sum_seq_len).unsqueeze(1),
                exist_tp1_label_assigned_pred_idxes
            ] = exist_tp1_label_assigned_pred_idxes != -1
            pred_assigned_to_exist_tp1_label = pred_assigned_to_exist_tp1_label[:, :-1]

            # case II: exist @ t but not exist @ t + 1, subcase 2: caused by sam error
            slot_disappear_due_to_sam_error = exist_t_not_exist_tp1 & pred_assigned_to_exist_tp1_label

        # need a clone to enable gradient computation
        assigned_pred_idxes = assigned_pred_idxes.clone()
        pred_assigned = pred_assigned.clone()
        slot_disappear_due_to_sam_error = slot_disappear_due_to_sam_error.clone()

        assert torch.all(assigned_pred_idxes < self.num_slots) and torch.all(assigned_pred_idxes >= 0), \
            f"assigned_pred_idxes must be in [0, {self.num_slots - 1}]"

        return assigned_pred_idxes, pred_assigned, slot_disappear_due_to_sam_error

    def get_pred_and_label_without_assignment(
        self,
        exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_exist_tp1, not_exist_t_not_exist_tp1,
        pred, label=None,
    ):
        # helper function to get prediction and label according to the assignment
        assert pred.shape == exist_t_exist_tp1.shape
        assert pred.ndim == 2 and pred.shape[1] == self.num_slots

        masks = {
            "case_I": exist_t_exist_tp1,
            "case_II": exist_t_not_exist_tp1,
            "case_III": not_exist_t_exist_tp1,
            "case_IV": not_exist_t_not_exist_tp1,
        }

        pred = {case_name: pred[mask] for case_name, mask in masks.items()}

        if label is None:
            return pred

        label = label.to(self.dtype)

        label = {case_name: label[mask] for case_name, mask in masks.items()}

        return pred, label

    def get_pred_and_label_with_assignment(
        self,
        exist_t_exist_tp1, exist_t_not_exist_tp1, not_exist_t_exist_tp1,
        assigned_pred_idxes, pred_assigned, slot_disappear_due_to_sam_error,
        pred, label=None,
        train_conf=False,
    ):
        assert pred.shape == exist_t_exist_tp1.shape
        assert pred.ndim == 2 and pred.shape[1] == self.num_slots

        assigned_pred = pred.gather(1, assigned_pred_idxes)

        # case I: exist @ t and exist @ t + 1
        case_I_mask = exist_t_exist_tp1

        # case II: exist @ t but not exist @ t + 1
        # subcase 1: natural and predictable, subcase 2: caused by sam error
        case_II_subcase1_mask = exist_t_not_exist_tp1 & ~slot_disappear_due_to_sam_error
        case_II_subcase2_mask = exist_t_not_exist_tp1 & slot_disappear_due_to_sam_error

        # case III: not exist @ t but exist @ t + 1
        # subcase 1: natural and predictable
        # subcase 2: natural but not so predictable
        # subcase 3: caused by sam error
        # not sure how to optimize subcases 1 and 2 yet
        # label at [b, j] is assigned to a prediction whose slot exists at t but not at t + 1
        use_pred_from_exist_t_not_exist_tp1_slots = exist_t_not_exist_tp1.gather(1, assigned_pred_idxes)
        case_III_subcase12_mask = not_exist_t_exist_tp1 & ~use_pred_from_exist_t_not_exist_tp1_slots
        case_III_subcase3_mask = not_exist_t_exist_tp1 & use_pred_from_exist_t_not_exist_tp1_slots

        masks = {
            "case_I": case_I_mask,
            "case_II_subcase1": case_II_subcase1_mask,
            "case_II_subcase2": case_II_subcase2_mask,
            "case_III_subcase12": case_III_subcase12_mask,
            "case_III_subcase3": case_III_subcase3_mask,
        }

        cased_pred = {case_name: assigned_pred[mask] for case_name, mask in masks.items()}
        # prediction not assigned to any valid label should be invisible and non-existent
        cased_pred["unassigned"] = pred[~pred_assigned]

        excluded_cases = []
        if not train_conf:
            excluded_cases.append("case_II_subcase2")
            if self.config.dynamics.pred.pred_appearing_slots:
                if not self.config.dynamics.pred.pred_appearing_slots_exist_visible:
                    excluded_cases.append("case_III_subcase12")
            else:
                excluded_cases.extend(["case_III_subcase12", "case_III_subcase3"])

        pred = {
            case_name: cased_pred_i for case_name, cased_pred_i in cased_pred.items()
            if case_name not in excluded_cases
        }

        if label is None:
            return pred

        assert label.shape == exist_t_exist_tp1.shape
        label = label.to(self.dtype)

        cased_label = {case_name: label[mask] for case_name, mask in masks.items()}
        # prediction not assigned to any valid label should be invisible and non-existent
        cased_label["unassigned"] = torch.zeros_like(cased_pred["unassigned"])

        label = {
            case_name: cased_label_i for case_name, cased_label_i in cased_label.items()
            if case_name not in excluded_cases
        }

        return pred, label

    # ============ Generation ============
    # TODO: modify mamba generation
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    @torch.inference_mode()
    def rollout(self, batch: Episode, random_static: bool = False, random_one_slot: bool = False):
        # warmup SSM for rollout
        bs, seq_len = batch.actions.shape[:2]
        n_warmup_steps = self.config.dynamics.rollout.n_warmup_steps
        n_rollout_steps = seq_len - n_warmup_steps

        warmup_inputs = Episode(**{
            k: v[:, :n_warmup_steps]
            for k, v in dataclasses.asdict(batch).items()
            if v is not None
        })
        if random_static:
            warmup_inputs = self.perturb_static(warmup_inputs, random_one_slot)

        step_rollout = self.reset_rollout(
            warmup_inputs,
            n_rollout_steps,
            deterministic=self.config.dynamics.rollout.deterministic,
        )
        step_rollouts = [step_rollout]

        # rollout SSM
        for i in range(n_warmup_steps, seq_len):
            step_rollout = self.step_rollout(
                step_rollout,
                actions=batch.actions[:, i:i+1],
                deterministic=self.config.dynamics.rollout.deterministic,
            )
            step_rollouts.append(step_rollout)

        # only keep the roll-out steps when computing loss
        batch = dataclasses.replace(batch, **{
            k: v[:, n_warmup_steps - 1:]
            for k, v in dataclasses.asdict(batch).items()
            if v is not None
        })

        # stack step_rollouts
        prediction = torch_cat_dataclasses_list(step_rollouts, dim=1)

        return batch, prediction

    def perturb_static(self, warmup_inputs, random_one_slot, random_prob=1.0):
        # mask invisible or non-existing slots to 0 for easier regression
        slots = warmup_inputs.slots
        last_slots_visible = warmup_inputs.slots_visible[:, -1]             # (bs, num_slots)
        last_slots = warmup_inputs.slots[:, -1]

        if self.disentangle_static_dynamic:
            slots[~warmup_inputs.slots_visible] = 0

            static_embeddings = self.proj_static(slots)
            last_static_embeddings = static_embeddings[:, -1]

            dynamic_embeddings, _, _ = self.compute_dynamic_embeddings(slots)

        is_slot_to_sample = last_slots_visible

        if random_one_slot:     # random pick one visible slot
            bs, _ = last_slots_visible.shape
            row_indices, col_indices = torch.where(last_slots_visible)
            counts = torch.bincount(row_indices, minlength=bs)

            # Create a mask to identify rows with at least one True value
            non_empty_rows = counts > 0

            # Generate random indices for selection within each row
            random_indices = torch.zeros(bs, dtype=torch.long, device=self.device)
            random_indices[non_empty_rows] = torch.randint(
                0, counts.max(), (non_empty_rows.sum(),), device=self.device
            ) % counts[non_empty_rows]

            # Map these random indices to the corresponding column indices
            random_col_indices = torch.zeros(bs, dtype=torch.long, device=self.device)
            random_col_indices[non_empty_rows] = col_indices[
                (torch.cumsum(counts, 0) - counts + random_indices)[non_empty_rows]
            ]

            # Construct the output tensor B
            is_slot_to_sample = torch.zeros_like(last_slots_visible)

            is_perturbed = torch.rand(bs, device=self.device) < random_prob
            batch_idxes = torch.arange(bs, device=self.device)[is_perturbed]
            random_col_indices = random_col_indices[is_perturbed]
            is_slot_to_sample[batch_idxes, random_col_indices] = True
            is_slot_to_sample[~non_empty_rows] = False

        # sample random static embeddings which will overwrite the static embeddings during model forward()
        slots_to_sample = last_slots[is_slot_to_sample]
        new_slots = self.slot_kmeans.sample_from_same_cluster(slots_to_sample)
        new_slots = torch.from_numpy(new_slots).to(**self.factory_kwargs)

        if self.disentangle_static_dynamic:
            new_static_embeddings = self.proj_static(new_slots)                         # (num_new_slots, slot_dim)

            last_static_embeddings[is_slot_to_sample] = new_static_embeddings

            seq_len = warmup_inputs.slots.shape[1]
            static_embeddings = repeat(last_static_embeddings, 'b s d -> b t s d', t=seq_len)

            slots = self.compute_slots_recon(dynamic_embeddings, static_embeddings)

        else:
            last_slots[is_slot_to_sample] = new_slots
            slots[:, -1] = last_slots

        warmup_inputs = dataclasses.replace(warmup_inputs, **{
            "slots": slots,
        })

        return warmup_inputs

    @torch.inference_mode()
    def reset_rollout(
        self,
        batch: Episode,
        n_rollout_steps: int,
        deterministic: bool = True,
    ):
        # warmup SSM for rollout
        n_warmup_steps = self.config.dynamics.rollout.n_warmup_steps

        step_rollout = super().reset_rollout(
            batch,
            max_length=n_warmup_steps + n_rollout_steps,
            deterministic=deterministic,
            cg=self.config.dynamics.rollout.cg,
            batch_size_multiplier=self.num_slots,
        )
        return step_rollout

    @torch.inference_mode()
    def step_rollout(self, step_rollout: Prediction, actions: torch.LongTensor, deterministic: bool = True):
        if self.config.dynamics.patch_as_slot:
            slots = None
            enc_feat = step_rollout.next_slots_pred
        else:
            slots = step_rollout.next_slots_pred
            enc_feat = None

        inputs = Episode(
            slots=slots,                                                    # (bs, 1, num_slots, slot_dim)
            enc_feat=enc_feat,                                              # (bs, 1, num_slots, slot_dim)
            slots_visible=step_rollout.next_slots_visible_pred,             # (bs, 1, num_slots)
            slots_exist=step_rollout.next_slots_exist_pred,                 # (bs, 1, num_slots)
            actions=actions,                                                # (bs, 1, )
        )
        step_rollout = super().step_rollout(
            inputs,
            deterministic=deterministic,
            cg=self.config.dynamics.rollout.cg,
            batch_size_multiplier=self.num_slots,
        )
        return step_rollout

    # ============ Save and Load ============
    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_PATH / checkpoint_path
        assert checkpoint_path.exists(), f"=> no dynamics checkpoint found at '{checkpoint_path}'"

        # open checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # verify config is the same
        checkpoint_dynamics_config = checkpoint["dynamics_config"]
        if self.config.dynamics.pred.disentangle_static_dynamic and not self.config.dynamics.training.update_pred:
            checkpoint_dynamics_config["pred"]["disentangle_static_dynamic"] = True

        assert are_dicts_equal(
            dataclasses.asdict(self.config.dynamics),
            checkpoint_dynamics_config,
            exclude_keys=["checkpoint_path", "training", "data_loading", "rollout"],
            only_use_common_keys=True,
        ), "config mismatch (see above)"
        assert are_dicts_equal(
            dataclasses.asdict(self.config.encoder),
            checkpoint["encoder_config"],
            keys=["slot_dim"],
        ), "config mismatch (see above)"

        # load model state dict
        model_state_dict = checkpoint["model"]

        # remove ddp prefix
        model_state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in model_state_dict.items()
        }
        msg = self.load_state_dict(model_state_dict, strict=False)
        print(f"=> dynamics loaded model from checkpoint: {checkpoint_path} with msg {msg}")

    def save_dict(self):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        return {
            "model": self.state_dict(),
            "dynamics_config": dataclasses.asdict(self.config.dynamics),
            "encoder_config": dataclasses.asdict(self.config.encoder),
        }
