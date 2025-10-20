from dataclasses import asdict, dataclass
from einops import reduce
from typing import Optional

import torch
import torch.nn.functional as F

from oc_utils.utils import pack_one


EPS = 1e-5


@dataclass
class LossBreakdown:
    entropy: Optional[torch.Tensor] = None
    perplexity: Optional[torch.Tensor] = None
    commitment: Optional[torch.Tensor] = None

    # for LFQ
    per_sample_entropy: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.entropy is not None:
            self.entropy = self.entropy.detach()
        if self.perplexity is not None:
            self.perplexity = self.perplexity.detach()
        if self.commitment is not None:
            self.commitment = self.commitment.detach()
        if self.per_sample_entropy is not None:
            self.per_sample_entropy = self.per_sample_entropy.detach()

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@torch.no_grad()
def compute_entropy_perplexity(
    indices: torch.Tensor,
    codebook_size: int,
):
    """
    Compute entropy of the indices.

    Args:
        indices: (..., c)
            - c: number of codebooks
        codebook_size: int

    Returns:
    """
    indices, _ = pack_one(indices, "* c")                               # (b, c)
    indices_onehot = F.one_hot(indices, codebook_size)                  # (b, c, n)

    avg_probs = indices_onehot.float().mean(dim=0)                      # (c, n)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS))
    perplexity = torch.exp(entropy)

    return entropy, perplexity


def entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + EPS, -1)

    avg_probs = reduce(probs, "... d -> d", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (batch_maximization_weight * avg_entropy)

    return sample_entropy, avg_entropy, loss
