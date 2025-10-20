import torch
import torch.nn.functional as F
import torch.distributed as dist


class LeCAM_EMA:
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

        if dist.is_initialized():
            self.all_reduce_fn = lambda x: dist.all_reduce(x, op=dist.ReduceOp.AVG)
        else:
            self.all_reduce_fn = lambda x: x

    def update(self, logits_real, logits_fake):
        logits_real = self.all_reduce_fn(logits_real)
        logits_fake = self.all_reduce_fn(logits_fake)
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real) * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake) * (1 - self.decay)

    def lecam_reg(self, real_pred, fake_pred):
        reg = torch.mean(F.relu(real_pred - self.logits_fake_ema).pow(2)) + \
              torch.mean(F.relu(self.logits_real_ema - fake_pred).pow(2))
        return reg


def log_sinkhorn_algorithm(cost_matrix, epsilon=0.1, num_iters=50):
    """
    Performs the Sinkhorn algorithm in log space for a batch of cost matrices.
    - cost_matrix: Tensor of shape (B, N, M), where
        B is batch size,
        N is number of ground truth labels, and
        M is number of predictions.
    - epsilon: Regularization parameter.
    - num_iters: Number of iterations for Sinkhorn normalization.
    Returns:
    - log_P: Logarithm of the doubly stochastic matrix representing soft assignment for each batch element (B, N, M).
    """
    B, N, M = cost_matrix.shape
    cost_invalid = cost_matrix == float("inf")
    log_u_ele_invalid = torch.all(cost_invalid, dim=2, keepdim=True)      # (B, N, 1)
    log_v_ele_invalid = torch.all(cost_invalid, dim=1, keepdim=True)      # (B, 1, M)

    log_K = -cost_matrix / epsilon

    # Initialize scaling vectors (log_u and log_v) in log space
    log_u = torch.zeros((B, N, 1), device=cost_matrix.device)
    log_v = torch.zeros((B, 1, M), device=cost_matrix.device)

    # Perform Sinkhorn iterations in log space
    for _ in range(num_iters):
        log_u = -torch.logsumexp(log_K + log_v, dim=2, keepdim=True)
        # avoid -inf + inf = nan when updating log_v
        log_u[log_u_ele_invalid] = 0

        log_v = -torch.logsumexp(log_K + log_u, dim=1, keepdim=True)
        # avoid -inf + inf = nan when updating log_u
        log_v[log_v_ele_invalid] = 0

    log_P = log_K + log_u + log_v
    assert not torch.isnan(log_P).any()

    return log_P


def complete_hard_assignment(cost):
    """
    Convert cost matrix (B, N, M) to a hard assignment using greedy matching.
    Returns:
    - hard_assignment: Tensor of shape (B, N),
        where each element represents the index of the prediction assigned to each ground truth label.
    """
    B, N, M = cost.shape
    ground_truth_invalid = torch.all(cost == float("inf"), dim=2)                       # (B, N)

    sorted_indices = torch.argsort(cost, dim=2)                                         # (B, N, M)

    hard_assignment = torch.full((B, N), -1, dtype=torch.long)
    for b in range(B):
        # Track assigned slots
        pred_assigned = torch.zeros(M, dtype=torch.bool)
        for i in range(N):
            if ground_truth_invalid[b, i]:
                continue

            for j in sorted_indices[b, i]:
                # If this slot is not yet assigned
                if not pred_assigned[j]:
                    hard_assignment[b, i] = j
                    pred_assigned[j] = True
                    break

    return hard_assignment