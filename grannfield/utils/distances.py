"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield
"""

import math

import numpy as np
import torch
from torch import nn


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter('coeff', nn.Parameter(coeff))
            self.register_parameter('offset', nn.Parameter(offset))
        else:
            self.register_buffer('coeff', coeff)
            self.register_buffer('offset', offset)

    def _initial_params(self):
        offset = torch.linspace(self.start, self.stop, self.num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, distances):
        distances = distances.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(distances, 2))


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs