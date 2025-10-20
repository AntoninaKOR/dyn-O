# modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py
# Copyright (c) 2023, Albert Gu, Tri Dao.
import gc
from dataclasses import dataclass, field, asdict, fields
from typing import Callable, Optional, Union
from einops import rearrange
from torch.distributions import Categorical, Bernoulli

import torch
from torch import Tensor

from oc_utils.utils import FloatTensor
from oc_utils.replay_buffer.data_format import Episode
from oc_utils.reward import symexp, compute_softmax_over_buckets


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class Prediction:
    next_slots_dist: FloatTensor
    next_slots_mode_logits: FloatTensor

    rewards_logits: FloatTensor
    terminations_logits: FloatTensor

    next_slots_visible_logits: Optional[FloatTensor] = None
    next_slots_exist_logits: Optional[FloatTensor] = None

    # sampled predictions
    next_slots_pred: Optional[FloatTensor] = None
    next_slots_visible_pred: Optional[torch.BoolTensor] = None
    next_slots_exist_pred: Optional[torch.BoolTensor] = None
    rewards_pred: Optional[FloatTensor] = None
    terminations_pred: Optional[torch.BoolTensor] = None

    # optional predictions for static-dynamic disentanglement training
    static_embeddings: Optional[FloatTensor] = None
    dynamic_embeddings: Optional[FloatTensor] = None
    next_dynamic_embeddings_dist: Optional[FloatTensor] = None
    next_dynamic_embeddings_pred: Optional[FloatTensor] = None

    def sample(self, deterministic: bool = False):
        if deterministic:
            next_slots_mode_idx = self.next_slots_mode_logits.argmax(dim=-1, keepdim=True)
            if self.next_slots_exist_logits is not None:
                self.next_slots_exist_pred = self.next_slots_exist_logits > 0
            if self.next_slots_visible_logits is not None:
                self.next_slots_visible_pred = self.next_slots_visible_logits > 0

            self.terminations_pred = self.terminations_logits > 0
        else:
            num_modes = self.next_slots_mode_logits.shape[-1]
            if num_modes == 1:
                next_slots_mode_idx = torch.zeros_like(self.next_slots_mode_logits, dtype=torch.long)
            else:
                next_slots_mode_idx = Categorical(logits=self.next_slots_mode_logits).sample()

            if self.next_slots_exist_logits is not None:
                self.next_slots_exist_pred = Bernoulli(logits=self.next_slots_exist_logits).sample().bool()
            if self.next_slots_visible_logits is not None:
                self.next_slots_visible_pred = Bernoulli(logits=self.next_slots_visible_logits).sample().bool()

            self.terminations_pred = Bernoulli(logits=self.terminations_logits).sample().bool()

        next_slots_pred = self.next_slots_dist.gather(
            -2,
            # (bs, seq_len, num_slots, 1) -> (bs, seq_len, num_slots, 1, slot_dim)
            next_slots_mode_idx.unsqueeze(dim=-1).expand(-1, -1, -1, -1, self.next_slots_dist.shape[-1]),
        )
        self.next_slots_pred = rearrange(next_slots_pred, 'b t s 1 d -> b t s d')

        if self.next_dynamic_embeddings_dist is not None:
            next_dynamic_embeddings_pred = self.next_dynamic_embeddings_dist.gather(
                -2,
                next_slots_mode_idx.unsqueeze(dim=-1).expand(-1, -1, -1, -1, self.next_dynamic_embeddings_dist.shape[-1]),
            )
            self.next_dynamic_embeddings_pred = rearrange(next_dynamic_embeddings_pred, 'b t s 1 d -> b t s d')

        self.rewards_pred = symexp(compute_softmax_over_buckets(self.rewards_logits))

        # only existing slots can be visible
        if self.next_slots_exist_pred is not None and self.next_slots_visible_pred is not None:
            self.next_slots_visible_pred[~self.next_slots_exist_pred] = False

        return self


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def reset_rollout(
        self,
        inputs: Episode,
        max_length,
        deterministic: bool = False,
        cg: bool = False,
        batch_size_multiplier: int = 1,
        **mixer_kwargs,
    ):
        """
        inputs: dict[str, Tensor], Tensor shapes: (batch_size, seq_len, item_shape)
        """
        batch_size, seqlen_og = inputs.actions.shape[:2]
        batch_size *= batch_size_multiplier
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                batch_size,
                seqlen_og,
                max_length,
                inputs,
                batch_size_multiplier=batch_size_multiplier,
            )
            self.inference_params = self._decoding_cache.inference_params
            self.inference_params.reset(max_length, batch_size)
        else:
            self.inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

        return GenerationMixin.step_rollout(self, inputs, deterministic=deterministic, batch_size_multiplier=batch_size_multiplier, **mixer_kwargs)

    @torch.inference_mode()
    def step_rollout(
        self,
        inputs: Episode,
        deterministic: bool = False,
        cg: bool = False,
        batch_size_multiplier: int = 1,
        **mixer_kwargs,
    ):
        """
        inputs: dict[str, Tensor], Tensor shapes: (batch_size, seq_len, item_shape)
        """

        def get_prediction(inputs_, inference_params):
            batch_size = inputs.actions.shape[0]
            device = next(self.parameters()).device
            decoding = inference_params.seqlen_offset > 0
            if decoding:
                position_ids = torch.full(
                    (batch_size * batch_size_multiplier, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=device,
                )
            else:
                position_ids = None
            if not cg or not decoding:
                predictions_ = self.compute_prediction_logits(
                    inputs_,
                    position_ids=position_ids,
                    num_last_tokens=1,
                    inference_params=inference_params,
                    **mixer_kwargs,
                )
            else:
                predictions_ = self._decoding_cache.run(
                    inputs_, position_ids, inference_params.seqlen_offset, **mixer_kwargs,
                )
            predictions_ = predictions_.sample(deterministic=deterministic)
            return predictions_

        assert self.inference_params.seqlen_offset < self.inference_params.max_seqlen, \
            f"Cannot generate more tokens, max length {self.inference_params.max_seqlen} reached in cache."

        predictions = get_prediction(inputs, self.inference_params)
        self.inference_params.seqlen_offset += inputs.actions.shape[1]
        return predictions


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    inputs: Episode,
    decoding_seqlens=(1,),
    dtype=None,
    n_warmups=2,
    batch_size_multiplier=1,
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size != cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, "allocate_inference_cache"), "CUDA graph decoding requires that the model has a method allocate_inference_cache"
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                inputs,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
                batch_size_multiplier=batch_size_multiplier,
            )

    def dispatch(inputs_, position_ids, seqlen):
        batch_size_, decoding_seqlen_ = inputs_.actions.shape[:2]
        return cache.callables[batch_size_ * batch_size_multiplier, decoding_seqlen_](inputs_, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, inference_params, batch_size, max_seqlen, inputs,
    decoding_seqlen=1, mempool=None, n_warmups=2, batch_size_multiplier=1,
):
    param = next(iter(model.parameters()))
    dtype, device = param.dtype, param.device
    assert batch_size % batch_size_multiplier == 0, "batch_size must be divisible by batch_size_multiplier"
    input_batch_size = batch_size // batch_size_multiplier
    inputs_ = Episode(**{
        k: torch.full((input_batch_size, decoding_seqlen, *v.shape[2:]), 0, dtype=v.dtype, device=device)
        for k, v in asdict(inputs).items()
        if isinstance(v, torch.Tensor)
    })
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            predictions = model.compute_prediction_logits(
                inputs_,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            )
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        predictions = model.compute_prediction_logits(
            inputs_,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        )

    def run(new_inputs, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        for field_ in fields(new_inputs):
            new_input_value = getattr(new_inputs, field_.name)
            if isinstance(new_input_value, torch.Tensor):
                old_input_value = getattr(inputs_, field_.name)
                if isinstance(old_input_value, torch.Tensor):
                    old_input_value.copy_(new_input_value)
                else:
                    setattr(inputs_, field_.name, new_input_value.clone())

        position_ids.copy_(new_position_ids)
        graph.replay()
        prediction_dict = {
            k: v.clone() if isinstance(v, Tensor) else v
            for k, v in asdict(predictions).items()
        }
        return Prediction(**prediction_dict)
    inference_params.seqlen_offset = seqlen_offset_og
    return run
