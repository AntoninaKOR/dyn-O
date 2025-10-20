"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""
from einops import rearrange
from typing import List
import numpy as np

import torch
import torch.nn as nn

from dynamics.mamba.quantization.utils import EPS, LossBreakdown, compute_entropy_perplexity
from oc_utils.utils import pack_one, unpack_one


def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()


def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()


class FiniteScalarQuantization(nn.Module):
    def __init__(
        self,
        codebook_levels: List[int],
        codebook_dim: int,
        num_codebooks: int = 1,
        preserve_symmetry: bool = False,
        noise_dropout: float = 0.0,
    ):
        super().__init__()

        _levels = torch.tensor(codebook_levels, dtype=torch.int64)
        self.register_buffer('_levels', _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + codebook_levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer('_basis', _basis, persistent=False)

        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = int(np.prod(codebook_levels))
        self.code_dim = len(codebook_levels)

        self.effective_codebook_dim = self.num_codebooks * self.code_dim

        if self.effective_codebook_dim != self.codebook_dim:
            self.project_in = nn.Linear(self.codebook_dim, self.effective_codebook_dim)
            self.project_out = nn.Linear(self.effective_codebook_dim, self.codebook_dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

    @property
    def indice_logits_shape(self):
        return (self.num_codebooks, self.codebook_size)

    def bound(self, z):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + EPS) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        bound_fn = self.symmetry_preserving_bound if self.preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # if using noise dropout, determine where to add a random offset elementwise
        if not self.training or self.noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, self.noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.code_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(torch.int64)

    def forward(self, z):
        """
        einstein notation
        b - batch
        c - number of codebooks
        d - codebook dim
        """

        assert z.shape[-1] == self.codebook_dim, f'expected dimension of {self.codebook_dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        # preprocess
        z, ps = pack_one(z, "* cd")                                                             # (b, c * d)

        # split out number of codebooks
        z = rearrange(z, "b (c d) -> b c d", c=self.num_codebooks)

        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)

        z_q = rearrange(z_q, 'b c d -> b (c d)')

        z_q = unpack_one(z_q, ps, "* cd")
        indices = unpack_one(indices, ps, "* c")

        z_q = self.project_out(z_q)

        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        entropy, perplexity = compute_entropy_perplexity(indices, self.codebook_size)

        return z_q, indices, loss, LossBreakdown(entropy=entropy, perplexity=perplexity)

    def get_codebook_entry(self, indices):
        """ Inverse of `codes_to_indices`. """

        codes = self._indices_to_codes(indices)
        codes = rearrange(codes, '... c d -> ... (c d)')
        codes = self.project_out(codes)

        return codes
