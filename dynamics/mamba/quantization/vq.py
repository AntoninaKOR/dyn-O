from einops import rearrange, repeat
from einx import get_at

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dynamics.mamba.quantization.utils import LossBreakdown, compute_entropy_perplexity
from oc_utils.utils import pack_one, unpack_one


class VectorQuantization(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 1,
        codebook_size: int = 8,
        codebook_dim: int = 64,
        l2_normalize: bool | None = False,
        ema_update: bool | None = False,
        ema_decay: float = 0.99,
        commit_weight: float = 0.25,
        codebook_weight: float = 0.25,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.l2_normalize = l2_normalize
        self.ema_update = ema_update
        self.ema_decay = ema_decay

        self.codebook_weight = codebook_weight
        self.commit_weight = commit_weight

        embedding = torch.zeros(self.num_codebooks, self.codebook_size, self.codebook_dim)
        embedding.uniform_(-1.0 / self.codebook_size * 5 , 1.0 / self.codebook_size * 5)

        if self.ema_update:
            self.register_buffer("embedding", embedding)
            self.register_buffer("ema_count", torch.ones(self.num_codebooks, self.codebook_size))
            self.register_buffer("ema_weight", self.embedding.clone())
        else:
            self.embedding = nn.Parameter(embedding)

        self.all_reduce_fn = dist.all_reduce if dist.is_initialized() else lambda x: x

    def forward(self, z):
        """
        args:
            z: (..., c * d) tensor
        returns:
            z_q: (..., c * d) tensor
            indices: (..., c) tensor
            loss: scalar tensor
            loss_breakdown: LossBreakdown

        einstein notations:
            b - batch
            c - number of codebooks
            n - codebook size
            d - codebook dim
        """
        # preprocess
        z, ps = pack_one(z, "* cd")                                                             # (b, c * d)

        # split out number of codebooks
        z = rearrange(z, "b (c d) -> c b d", c=self.num_codebooks)

        # compute distances
        if self.l2_normalize:
            z_norm = torch.nn.functional.normalize(z)                                           # (c, b, d)
            embedding_norm = torch.nn.functional.normalize(self.embedding)                      # (c, n, d)

            d = -torch.einsum("c b d, c n d -> c b n", z_norm, embedding_norm)                  # (c, b, n)
        else:
            d = torch.sum(z ** 2, dim=-1, keepdim=True) + \
                torch.sum(self.embedding ** 2, dim=-1).unsqueeze(-2) - \
                2 * torch.einsum("c b d, c n d -> c b n", z, self.embedding)                    # (c, b, n)

        # quantize
        indices = torch.argmin(d, dim=-1)                                                       # (c, b)
        indices_onehot = F.one_hot(indices, self.codebook_size).to(torch.get_default_dtype())   # (c, b, n)

        z_q = torch.gather(self.embedding, 1, repeat(indices, "c b -> c b d", d=self.codebook_dim))

        # codebook update
        commit_loss = F.mse_loss(z_q.detach(), z)
        loss = commit_loss
        loss = self.commit_weight * loss

        if self.ema_update:
            if self.training:
                with torch.no_grad():
                    # update the codebook using exponential moving average
                    code_count = torch.sum(indices_onehot, dim=1)
                    self.all_reduce_fn(code_count)
                    self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * code_count

                    z_sum = torch.bmm(rearrange(indices_onehot, "c b n -> c n b"), z)               # (c, n, d)
                    self.all_reduce_fn(z_sum)
                    self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * z_sum

                    self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        else:
            # update the codebook using the codebook loss
            codebook_loss = F.mse_loss(z_q, z.detach())
            loss = loss + self.codebook_weight * codebook_loss

        z_q = z + (z_q - z).detach()

        # entropy for logging
        entropy, perplexity = compute_entropy_perplexity(
            rearrange(indices, "c b -> b c"),
            self.codebook_size,
        )

        # postprocess
        z_q = rearrange(z_q, "c b d -> b (c d)")
        indices = rearrange(indices, "c b -> b c")

        z_q = unpack_one(z_q, ps, "* d")
        indices = unpack_one(indices, ps, "* c")

        return z_q, indices, loss, LossBreakdown(commitment=commit_loss, entropy=entropy, perplexity=perplexity)

    def get_codebook_entry(self, indices):
        """
        args:
            indices: (..., c) tensor
        returns:
            z_q: (..., c * d) tensor

        einstein notations:
            b - batch
            c - number of codebooks
            n - codebook size
            d - codebook dim
        """

        z_q = get_at('c [n] d, ... c -> ... c d', self.embedding, indices)
        z_q = rearrange(z_q, "... c d -> ... (c d)")

        return z_q
