from typing import Optional
from dataclasses import dataclass, fields
import torch

from oc_utils.utils import FloatTensor


@dataclass
class Episode:
    actions: torch.LongTensor                                   # (bs, seq_len)

    enc_feat: Optional[FloatTensor] = None                      # (bs, seq_len, num_tokens, token_dim)
    slots: Optional[FloatTensor] = None                         # (bs, seq_len, num_slots, slot_dim)
    slots_visible: Optional[torch.BoolTensor] = None            # (bs, seq_len, num_slots)
    slots_exist: Optional[torch.BoolTensor] = None              # (bs, seq_len, num_slots)

    rewards: Optional[FloatTensor] = None                       # (bs, seq_len)
    terminations: Optional[torch.BoolTensor] = None             # (bs, seq_len)
    truncations: Optional[torch.BoolTensor] = None              # (bs, seq_len)

    rewards_weights: Optional[FloatTensor] = None               # (bs, seq_len)
    terminations_weights: Optional[FloatTensor] = None          # (bs, seq_len)

    # padding info
    padding_mask: Optional[torch.BoolTensor] = None             # (bs, seq_len)

    # for static-dynamic disentanglement
    temporal_random_idxes: Optional[torch.LongTensor] = None    # (bs, seq_len, num_slots)
    slot_obj_id: Optional[torch.LongTensor] = None              # (bs, seq_len, num_slots)
    static_embeddings: Optional[FloatTensor] = None             # (bs, num_slots, embedding_dim)
    dynamic_embeddings: Optional[FloatTensor] = None            # (bs, num_slots, embedding_dim)

    # for visualization
    observations: Optional[FloatTensor] = None                  # (bs, seq_len, 3, H, W)
    sam_masks: Optional[torch.LongTensor] = None                # (bs, seq_len, H, W)

    def pin_memory(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.pin_memory())

        return self

    @property
    def seq_len(self):
        return self.actions.shape[1]

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice))
        if self.static_embeddings is not None:
            raise NotImplementedError("Not implemented")
        if self.dynamic_embeddings is not None:
            raise NotImplementedError("Not implemented")

        sliced_attrs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                assert value.shape[1] == self.seq_len, f"Attribute {field.name} has invalid shape {value.shape}"
                sliced_attrs[field.name] = value[:, idx]

        return Episode(**sliced_attrs)
