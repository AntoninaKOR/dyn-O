import torch


@torch.no_grad()
def compute_lambda_returns(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    ends: torch.LongTensor,
    value_bootstrap: torch.FloatTensor,
    gamma: float,
    lambda_: float
) -> torch.FloatTensor:
    assert rewards.ndim == 2
    assert rewards.size() == values.size() == ends.size()
    assert value_bootstrap.ndim == 1 and value_bootstrap.size(0) == rewards.size(0)

    lambda_returns = rewards + ends.logical_not() * gamma * (1 - lambda_) * torch.cat((values[:, 1:], value_bootstrap.unsqueeze(1)), dim=1)

    last = value_bootstrap
    for t in list(range(rewards.size(1)))[::-1]:
        lambda_returns[:, t] += ends[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns


def compute_mask_after_first_done(ends: torch.LongTensor) -> torch.BoolTensor:
    assert ends.ndim == 2
    first_one_index = torch.argmax(ends, dim=1)
    mask = torch.arange(ends.size(1), device=ends.device).unsqueeze(0) <= first_one_index.unsqueeze(1)
    mask = torch.logical_or(mask, ends.sum(dim=1, keepdim=True) == 0)

    return mask
