"""
Credits to https://github.com/karpathy/minGPT
"""

from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, **mha_cfg) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            batch_first=True,
            **mha_cfg,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.ln1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_blocks: int, **mha_cfg) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionBlock(d_model, **mha_cfg) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
