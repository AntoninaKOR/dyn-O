# modified from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/sim_vq.py
from typing import List, Optional
from einops import rearrange
from einx import get_at

import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

from dynamics.mamba.quantization.utils import LossBreakdown, compute_entropy_perplexity
from oc_utils.utils import pack_one, unpack_one


class SimpleVectorQuantization(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        codebook_transform_mlp_multi: Optional[List[int]] = None,
        rotation_trick: bool = False,
        input_to_quantize_commit_loss_weight: float = 0.25,
        commit_weight: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size

        frozen_codebook = torch.randn(codebook_size, codebook_dim) * (codebook_dim ** -0.5)

        # the codebook is actually implicit from a linear layer from frozen gaussian or uniform

        if codebook_transform_mlp_multi is None:
            codebook_transform = nn.Linear(codebook_dim, codebook_dim, bias=False)
        else:
            layers = []
            codebook_transform_mlp = [multi * codebook_dim for multi in codebook_transform_mlp_multi]
            for in_dim, out_dim in zip([codebook_dim] + codebook_transform_mlp[:-1], codebook_transform_mlp):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(codebook_transform_mlp[-1], codebook_dim))
            codebook_transform = nn.Sequential(*layers)

        self.code_transform = codebook_transform
        self.register_buffer('frozen_codebook', frozen_codebook)

        # whether to use rotation trick from Fifty et al.
        # https://arxiv.org/abs/2410.06424
        self.rotation_trick = rotation_trick

        # commit loss weighting - weighing input to quantize a bit less is crucial for it to work
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight

        # total commitment loss weight
        self.commit_weight = commit_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def forward(self, z):
        """
        args:
            z: (..., d) tensor
        returns:
            z_q: (..., d) tensor
            indices: (..., 1) tensor
            loss: scalar tensor
            loss_breakdown: LossBreakdown

        einstein notations:
            b - batch
            n - codebook size
            d - codebook dim
        """
        z, ps = pack_one(z, '* d')                                                          # (b, d)

        implicit_codebook = self.codebook                                                   # (n, d)

        with torch.no_grad():
            dist = (
                torch.sum(z ** 2, dim=-1, keepdim=True) +
                torch.sum(implicit_codebook ** 2, dim=-1).unsqueeze(-2) -
                2 * torch.einsum("b d, n d -> b n", z, implicit_codebook)
            )                                                                               # (b, n)
            indices = dist.argmin(dim=-1)                                                   # (b)
            indices_onehot = F.one_hot(indices, self.codebook_size)                         # (b, n)
            indices_onehot = indices_onehot.to(torch.get_default_dtype())                   # (b, n)

        # select codes
        z_q = get_at('[n] d, b -> b d', implicit_codebook, indices)                         # (b, d)

        # commit loss and straight through, as was done in the paper
        commitment_loss = F.mse_loss(z.detach(), z_q)
        loss = (
            commitment_loss * self.commit_weight +
            F.mse_loss(z, z_q.detach()) * self.input_to_quantize_commit_loss_weight
        )

        if self.rotation_trick:
            # rotation trick from @cfifty
            z_q = rotate_to(z, z_q)
        else:
            z_q = (z_q - z).detach() + z                                                    # (b, d)

        # entropy for logging
        entropy, perplexity = compute_entropy_perplexity(
            rearrange(indices, 'b -> b 1'),
            self.codebook_size,
        )

        z_q = unpack_one(z_q, ps, '* d')
        indices = rearrange(indices, 'b -> b 1')
        indices = unpack_one(indices, ps, '* c')

        return z_q, indices, loss, LossBreakdown(commitment=commitment_loss, entropy=entropy, perplexity=perplexity)

    def get_codebook_entry(
        self,
        indices
    ):
        frozen_codes = get_at('[n] d, b ... -> b ... d', self.frozen_codebook, indices)
        z_q = self.code_transform(frozen_codes)

        return z_q
