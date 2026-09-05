"""The two small building blocks GRAFT's heads are made of.

Vendored verbatim (by AST source range, see tools/extract.py) from the research
tree's `network/modules.py`, which additionally imports DINOv2 and
`transformers` for encoders GRAFT does not use.
"""
import itertools
from typing import Sequence

import torch
import torch.nn as nn


class LearnableFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, concat_input: bool = True):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError('out_features must be even (sin and cos in pairs)')
        self.linear = nn.Linear(in_features, out_features // 2, bias=False)
        self.concat_input = concat_input
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=12)

    def forward(self, inp):
        x = self.linear(inp)
        out = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if self.concat_input:
            out = torch.cat([inp, out], dim=-1)
        return out


class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int], activation=nn.ReLU):
        nn.Sequential.__init__(self,
            *itertools.chain(*[
                (nn.Linear(dim_in, dim_out), activation(inplace=True) if isinstance(activation, nn.ReLU) else activation())
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
            ]),
            nn.Linear(dims[-2], dims[-1]),
        )
