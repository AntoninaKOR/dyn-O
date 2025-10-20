import torch.nn as nn


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, norm=True, dropout=0., act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features) if norm else nn.Identity()
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, final_act=None, dropout=0., norm=True):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0), norm=norm))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=final_act, norm=norm) if final_act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)
