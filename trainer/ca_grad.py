import numpy as np
from scipy.optimize import minimize

import torch


def cagrad(grads, alpha=0.5, rescale=True):
    new_grads = {}
    with torch.no_grad():
        for group_name, group_grad in grads.items():
            group_grad = group_grad[:, torch.any(group_grad != 0, dim=0)]  # remove tasks with zero gradients

            GG = group_grad.t().mm(group_grad).cpu()  # [num_tasks, num_tasks]
            g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

            num_tasks = group_grad.size(1)
            x_start = np.ones(num_tasks) / num_tasks
            bnds = tuple((0, 1) for x in x_start)
            cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
            A = GG.numpy()
            b = x_start.copy()
            c = (alpha * g0_norm + 1e-8).item()

            def objfn(x):
                return x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1) + \
                    c * np.sqrt(x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

            res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
            w_cpu = res.x
            ww = torch.tensor(w_cpu, dtype=group_grad.dtype, device=group_grad.device)
            gw = (group_grad * ww.view(1, -1)).sum(1)
            gw_norm = gw.norm()
            lmbda = c / (gw_norm + 1e-8)
            g = group_grad.mean(1) + lmbda * gw
            if rescale:
                g = g / g.norm() * group_grad.mean(1).norm()

            new_grads[group_name] = g * num_tasks

    return new_grads


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    with torch.no_grad():
        for group_name, module_group in m.shared_modules().items():
            grad_dim = grad_dims[group_name]
            grad = grads[group_name]

            grad[:, task].fill_(0.0)
            cnt = 0
            for mm in module_group:
                for p in mm.parameters():
                    if p.grad is not None:
                        start = grad_dim[cnt]
                        end = grad_dim[cnt + 1]
                        grad[start:end, task].copy_(p.grad.view(-1))
                    cnt += 1


def overwrite_grad(m, new_grads, grad_dims):
    with torch.no_grad():
        for group_name, module_group in m.shared_modules().items():
            grad_dim = grad_dims[group_name]
            new_grad = new_grads[group_name]

            cnt = 0
            for mm in module_group:
                for param in mm.parameters():
                    start = grad_dim[cnt]
                    end = grad_dim[cnt + 1]
                    this_grad = new_grad[start:end].contiguous().view(param.data.size())
                    param.grad = this_grad
                    cnt += 1
