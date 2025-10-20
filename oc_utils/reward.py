import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    x.clamp_(x_min, x_max - 1e-5)
    buckets = torch.linspace(x_min, x_max, num_buckets, device=x.device)
    k = torch.searchsorted(buckets, x) - 1
    values = torch.stack((buckets[k + 1] - x, x - buckets[k]), dim=-1) / (buckets[k + 1] - buckets[k]).unsqueeze(-1)
    two_hots = torch.scatter(x.new_zeros(*x.size(), num_buckets), dim=-1, index=torch.stack((k, k + 1), dim=-1), src=values)

    return two_hots


def compute_softmax_over_buckets(logits: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    buckets = torch.linspace(x_min, x_max, num_buckets, device=logits.device)
    probs = F.softmax(logits, dim=-1)

    return probs @ buckets