import dataclasses
from typing import Optional, List, Union, Literal
from functools import partial
from contextlib import nullcontext

try:
    import wandb
    wandb_available = True
except:
    wandb_available = False

import os
import glob
import mlflow
import numpy as np
from mlflow.client import MlflowClient
from mlflow.entities import Metric
from mlflow.utils.time import get_current_time_millis

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.world_model_env import WorldModelEnv
from trainer.ca_grad import grad2vec, cagrad, overwrite_grad
from oc_utils.utils import to_tensor


@dataclasses.dataclass
class DataMisc:
    dataloader: DataLoader
    data_iter: iter
    num_iters: int = 0
    sampler: Optional[DistributedSampler] = None


class Trainer:
    def __init__(self, config, encoder, dynamics, policy, rb):
        self.config = config
        self.encoder = encoder
        self.dynamics = dynamics
        self.policy = policy
        self.rb = rb

        self.dynamics_unwrapped = dynamics.module if isinstance(dynamics, DDP) else dynamics
        self.encoder_unwrapped = encoder.module if isinstance(encoder, DDP) else encoder
        self.policy_unwrapped = policy.module if isinstance(policy, DDP) else policy

        self.world_model_env = WorldModelEnv(
            config,
            self.encoder_unwrapped,
            self.dynamics_unwrapped,
        )

        # optimizer
        self.dynamics_optim = self.dynamics_unwrapped.configure_optimizers()

        if config.update_policy:
            self.policy_optim = self.policy_unwrapped.configure_optimizers()
            if isinstance(self.policy_optim, torch.optim.Optimizer):
                self.policy_optim = {"default": self.policy_optim}

        # only used for offline training, created when needed
        self.dataloader_dict = {}

        if self.config.dynamics.training.use_ca_grad:
            self.ca_grad_cache = None

    def update_dynamics(self, step):

        load_obs = self.config.dynamics.training.pred_obs
        load_enc_feat = self.config.dynamics.training.pred_enc_feat or self.config.dynamics.patch_as_slot
        load_slot = not self.config.dynamics.patch_as_slot

        disentangle_static_dynamic = self.config.dynamics.pred.disentangle_static_dynamic

        train_modes = self.dynamics_optim.keys()

        if "prediction" in train_modes:
            while True:
                pred_batch = self.sample_episode_from_buffer(
                    mode="dynamics",
                    training=True,
                    batch_size=self.config.dynamics.training.batch_size,
                    seq_len=self.config.dynamics.training.pred_seq_len,
                    load_obs=load_obs,
                    load_enc_feat=load_enc_feat,
                    load_slot=load_slot,
                    patch_as_slot=self.config.dynamics.patch_as_slot,
                    termination_prob=self.config.dynamics.training.termination_prob,
                )
                if not pred_batch.padding_mask[:, 1:].all():     # skip batch whose labels are all padded
                    break

        if disentangle_static_dynamic:
            while True:
                disentangle_batch = self.sample_episode_from_buffer(
                    mode="dynamics",
                    training=True,
                    batch_size=self.config.dynamics.training.batch_size,
                    seq_len=self.config.dynamics.training.disentangle_seq_len,
                    load_obs=load_obs,
                    load_enc_feat=load_enc_feat,
                    load_slot=load_slot,
                    patch_as_slot=self.config.dynamics.patch_as_slot,
                    termination_prob=self.config.dynamics.training.termination_prob,
                )
                if not disentangle_batch.padding_mask[:, 1:].all():     # skip batch whose labels are all padded
                    break

        self.dynamics.train()
        self.dynamics_unwrapped.set_scheduler_step(step)

        logging = {}

        n_training_steps = {
            "static": self.config.dynamics.training.n_static_steps,
            "dynamic": self.config.dynamics.training.n_dynamic_steps,
            "dynamic_disentanglement": self.config.dynamics.training.n_dynamic_disentanglement_steps,
            "discriminator": self.config.dynamics.training.n_discriminator_steps,
            "prediction": 1,
        }

        no_sync = self.config.distributed and disentangle_static_dynamic and self.config.dynamics.training.use_ca_grad

        with (
            self.dynamics.no_sync()
            if no_sync else
            nullcontext()
        ):

            for train_mode in train_modes:

                if train_mode == "prediction":
                    batch = pred_batch
                else:
                    batch = disentangle_batch

                if not disentangle_static_dynamic and train_mode != "prediction":
                    continue

                if train_mode == "discriminator" and self.config.dynamics.training.dynamic_adversarial_loss_weight_schedule.end_value <= 0:
                    continue

                self.set_requires_grad(self.dynamics, self.dynamics_optim[train_mode])

                for _ in range(n_training_steps[train_mode]):

                    self.dynamics.zero_grad()

                    if train_mode == "prediction":
                        _, _, loss, mode_logging = self.dynamics(batch, mode=train_mode)
                    else:
                        loss, mode_logging = self.dynamics(batch, mode=train_mode)

                    if isinstance(loss, dict):
                        assert len(loss) > 1
                        assert no_sync
                        assert train_mode == "dynamic"

                        # cache for CA grad
                        num_objectives = len(loss)
                        grad_dims, grads = self.get_ca_grad_cache(num_objectives)

                        for i, (loss_i_name, loss_i) in enumerate(loss.items()):
                            retain_graph = (i < num_objectives - 1)
                            loss_i.backward(retain_graph=retain_graph)
                            self.sync_gradients(self.dynamics_unwrapped.shared_modules())

                            grad2vec(self.dynamics_unwrapped, grads, grad_dims, i)
                            self.dynamics_unwrapped.zero_grad_shared_modules()

                        new_grads = cagrad(
                            grads,
                            self.config.dynamics.training.ca_grad_alpha,
                            rescale=self.config.dynamics.training.ca_grad_rescale,
                        )

                        # logging gradient cosine similarity
                        for group_name in grads:
                            group_grad = grads[group_name]
                            new_grad = new_grads[group_name]

                            grads_norm = torch.norm(group_grad, dim=0, keepdim=True)
                            grads_normed = group_grad / (grads_norm + 1e-8)

                            grads_cos_sim = torch.einsum("ij,ik->jk", grads_normed, grads_normed)

                            new_grad_norm = torch.norm(new_grad, dim=0)
                            new_grad_normed = new_grad / (new_grad_norm + 1e-8)

                            new_grad_cos_sim = torch.einsum("i,ik->k", new_grad_normed, grads_normed)

                            logging[f"grad_norm/{group_name}/new_grad_norm"] = new_grad_norm

                            for i, loss_i_name in enumerate(loss.keys()):
                                logging[f"grad_cos_sim/{group_name}/{loss_i_name}/new_grad"] = new_grad_cos_sim[i]

                                logging[f"grad_norm/{group_name}/{loss_i_name}"] = grads_norm[0, i]
                                for j, loss_j_name in enumerate(loss.keys()):
                                    if i >= j:
                                        continue
                                    logging[f"grad_cos_sim/{group_name}/{loss_i_name}/{loss_j_name}"] = grads_cos_sim[i, j]

                        overwrite_grad(self.dynamics_unwrapped, new_grads, grad_dims)

                    else:
                        loss.backward()

                    if no_sync:
                        self.sync_gradients(self.dynamics)

                    if self.config.dynamics.training.grad_norm_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.config.dynamics.training.grad_norm_clip)
                        logging[f"grad_norm/{train_mode}"] = grad_norm

                    self.dynamics_optim[train_mode].step()

                    self.dynamics.zero_grad()

                    logging.update(mode_logging)

        self.logging(logging, "dynamics", step)

    @staticmethod
    def set_requires_grad(model, optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]):
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        optimized_params = {
            p
            for optimizer in optimizers
            for group in optimizer.param_groups
            for p in group['params']
        }
        for param in model.parameters():
            param.requires_grad = param in optimized_params

    def sync_gradients(self, module):
        # manually synchronize gradients across processes
        if self.config.distributed:
            if isinstance(module, list):
                for module_ in module:
                    self.sync_gradients(module_)
            elif isinstance(module, dict):
                for module_ in module.values():
                    self.sync_gradients(module_)
            elif isinstance(module, torch.nn.Module):
                for param in module.parameters():
                    if param.grad is not None:
                        grad = param.grad / self.config.world_size
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        param.grad = grad
            else:
                raise ValueError(f"Unsupported module type: {type(module)}")

    @torch.inference_mode()
    def rollout_dynamics(self, split, step, rollout_kwargs):
        assert split in ["train", "valid"]

        load_enc_feat = self.config.dynamics.training.pred_enc_feat or self.config.dynamics.patch_as_slot
        load_slot = not self.config.dynamics.patch_as_slot

        while True:
            batch = self.sample_episode_from_buffer(
                mode="dynamics",
                training=True if split == "train" else False,
                batch_size=self.config.dynamics.training.batch_size,
                load_obs=True,
                load_enc_feat=load_enc_feat,
                load_slot=load_slot,
                seq_len=self.config.dynamics.rollout.n_warmup_steps + self.config.logging.rollout_viz_steps[-1],
                patch_as_slot=self.config.dynamics.patch_as_slot,
                termination_prob=-1,
            )
            seq_len = batch.padding_mask.shape[1]
            if seq_len > self.config.dynamics.rollout.n_warmup_steps:
                break

        self.dynamics.eval()

        inputs = {}
        predictions = {}

        for k, kwargs in rollout_kwargs.items():
            batch_modified, prediction, _, logging = self.dynamics(batch, **kwargs)

            inputs[k] = batch_modified
            predictions[k] = prediction

            if k == "rollout":
                self.logging(logging, f"dynamics_rollout_{split}", step)

        return inputs, predictions

    def update_policy(self, step):
        if self.config.policy_type == "actor_critic":
            if self.config.policy.training.use_imagination:
                seq_len = self.config.dynamics.rollout.n_warmup_steps
            else:
                seq_len = self.config.policy.world_model_env.n_rollout_steps + 1
        elif self.config.policy_type == "bc":
            seq_len = 1
        else:
            raise ValueError(f"Unsupported policy type: {self.config.policy_type}")

        load_enc_feat = self.config.policy.world_model_env.obs_mode == "enc_feat" or self.config.dynamics.patch_as_slot

        while True:
            batch = self.sample_episode_from_buffer(
                mode="policy",
                training=True,
                batch_size=self.config.policy.training.batch_size,
                load_obs=False,
                patch_as_slot=self.config.dynamics.patch_as_slot,
                seq_len=seq_len,
                load_enc_feat=load_enc_feat,
                load_slot=self.config.policy.world_model_env.obs_mode == "slots",
                termination_prob=-1,
            )
            seq_len = batch.padding_mask.shape[1]
            if seq_len == seq_len:
                break

        self.encoder.eval()
        self.dynamics.eval()
        self.policy.train()
        losses, logging = self.policy(batch=batch, world_model_env=self.world_model_env)

        if isinstance(losses, torch.Tensor):
            losses = {"default": losses}

        # update
        for k in losses:
            loss = losses[k]
            optim = self.policy_optim[k]

            self.policy.zero_grad()
            loss.backward()
            if self.config.policy.training.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.policy.training.grad_norm_clip)
            optim.step()

        self.logging(logging, "policy", step)

    def sample_episode_from_buffer(
        self,
        mode: str,
        training: bool,
        batch_size: int,
        seq_len: int = -1,
        load_obs: bool = False,
        load_sam_mask: bool = False,
        load_enc_feat: bool = False,
        load_slot: bool = False,
        patch_as_slot: bool = False,
        termination_prob: float = 0.0,
        **dataloader_kwargs,
    ):
        """
        :param mode:
        :param training:
        :param batch_size:
        :param seq_len: -1 uses all steps in the episode, otherwise sample an seq_len chunk
        :param load_obs: load rgb observations
        :param load_sam_mask: load sam mask
        :param load_enc_feat: load patch-level encoder-extracted features
        :param load_slot: load slots
        :param patch_as_slot: use patch-level encoder features instead object-level slots (for ablation studies)
        :param termination_prob:
            when seq_len != -1, the probability for a sampled chunk to have termination = True
            notice that such a chunk could have length < seq_len
            use > 0 value to avoid the sampled data to have too few termination = True labels during training
        :param dataloader_kwargs:
        :return:
            slots: (batch_size, seq_len, num_slots, slot_dim)
            slots_visible, slots_exist: (batch_size, seq_len, num_slots)
            actions: (batch_size, seq_len)
        """

        if self.config.offline:
            # initialize dataloader
            arguments = {
                "mode": mode,
                "split": "train" if training else "valid",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "load_obs": load_obs,
                "load_sam_mask": load_sam_mask,
                "load_enc_feat": load_enc_feat,
                "load_slot": load_slot,
                "patch_as_slot": patch_as_slot,
                "termination_prob": termination_prob,
            }

            keys = tuple(arguments.values())
            if keys not in self.dataloader_dict:
                dataset = self.rb.as_episode_dataset(**{
                    k: v for k, v in arguments.items()
                    if k not in ["mode", "batch_size"]
                })

                if training:
                    if self.config.distributed:
                        sampler = DistributedSampler(
                            dataset,
                            num_replicas=self.config.world_size,
                            rank=self.config.rank,
                            shuffle=True,
                        )
                        sampler.set_epoch(0)
                        shuffle = None
                        batch_size = batch_size // self.config.world_size
                    else:
                        sampler = None
                        shuffle = True
                else:
                    sampler = None
                    shuffle = False

                dataloader = DataLoader(
                    dataset,
                    sampler=sampler,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=partial(dataset.collate_fn),
                    num_workers=0,
                    pin_memory=True,
                    **dataloader_kwargs,
                )

                self.dataloader_dict[keys] = DataMisc(
                    sampler=sampler,
                    dataloader=dataloader,
                    data_iter=iter(dataloader),
                    num_iters=0,
                )

            data_misc = self.dataloader_dict[keys]

            try:
                batch = next(data_misc.data_iter)
                data_misc.num_iters += 1
                if data_misc.num_iters % len(data_misc.dataloader) == 0:
                    if data_misc.sampler is not None:
                        data_misc.sampler.set_epoch(data_misc.num_iters // len(data_misc.dataloader))
                    data_misc.data_iter = iter(data_misc.dataloader)
            except StopIteration:
                raise ValueError('should not reach here, data_iter should not be reset')

        else:
            raise NotImplementedError

        batch = to_tensor(batch, device=self.config.device, non_blocking=True)

        return batch

    def get_ca_grad_cache(self, num_objectives):
        if self.ca_grad_cache is None:
            grad_dims, grads = {}, {}
            for group_name, module_group in self.dynamics_unwrapped.shared_modules().items():
                grad_dim = [0]
                for mm in module_group:
                    for param in mm.parameters():
                        grad_dim.append(param.data.numel())
                grad_dim = np.cumsum(grad_dim)
                grad = torch.zeros(grad_dim[-1], num_objectives, dtype=self.config.dtype, device=self.config.device)

                grad_dims[group_name] = grad_dim
                grads[group_name] = grad
            self.ca_grad_cache = (grad_dims, grads)
        return self.ca_grad_cache

    # ========================== Log / Save / Load ==========================
    def logging(self, logging_output, prefix, step):
        if self.config.rank == 0 and step % self.config.logging.training_log_interval == 0:
            logging_output = {
                f"{prefix}/{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in logging_output.items()
            }

            if wandb_available:
                wandb.define_metric("step")
                for k in logging_output.keys():
                    wandb.define_metric(k, step_metric="step")

                wandb.log({**logging_output, "step": step})
            else:
                ts = get_current_time_millis()
                metrics = [
                    Metric(k, v, ts, step)
                    for k, v in logging_output.items()
                ]
                MlflowClient().log_batch(mlflow.active_run().info.run_id, metrics, synchronous=False)

    def dynamics_save_dict(self, global_step):
        save_dict = self.dynamics_unwrapped.save_dict()
        save_dict["optimizer"] = {k: v.state_dict() for k, v in self.dynamics_optim.items()}
        save_dict["global_step"] = global_step
        return save_dict

    def policy_save_dict(self, global_step):
        save_dict = self.policy_unwrapped.save_dict()
        save_dict["optimizer"] = {k: v.state_dict() for k, v in self.policy_optim.items()}
        save_dict["global_step"] = global_step
        return save_dict

    def load_last_checkpoint(self):
        if self.config.ckpt_path is None:
            return 1
        global_step = 1
        assert not (self.config.update_dynamics and self.config.update_policy), \
            "current loading assumes only one of dynamics and policy is updated"
        if self.config.update_dynamics:
            global_step = self.load_last_module_checkpoint(self.dynamics_unwrapped, self.dynamics_optim, "dynamics_*.pt")
        if self.config.update_policy:
            global_step = self.load_last_module_checkpoint(self.policy_unwrapped, self.policy_optim, "policy_*.pt")
        return global_step

    def load_last_module_checkpoint(self, module, optimizer, ckpt_pattern):
        ckpt_list = glob.glob(str(self.config.ckpt_path / ckpt_pattern))
        global_step = 1
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    ckpt_path = ckpt_list[-1]
                    module.load_checkpoint(ckpt_path)
                    ckpt = torch.load(ckpt_path)
                    optimizer_state_dict = ckpt["optimizer"]
                    if isinstance(optimizer, dict):
                        for k, v in optimizer.items():
                            v.load_state_dict(optimizer_state_dict[k])
                    else:
                        optimizer.load_state_dict(optimizer_state_dict)
                    global_step = ckpt["global_step"]
                    break
                except:
                    ckpt_list.pop()
        return global_step
