"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

stpayu/grannfield: https://github.com/stpayu/grannfield

---

This code has been modified from the original version at
Open-Catalyst-Project/ocp: https://github.com/Open-Catalyst-Project/ocp
"""

import torch
from torch_geometric.nn import SchNet, radius_graph
from torch_scatter import scatter

from grannfield.utils.utils import (
    get_pbc_distances,
    radius_graph_pbc
)


class SchNetWrap(SchNet):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_neighbors=50,
        readout="add",
        reduce="sum",
        offset_degree=1
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.offset_degree = offset_degree
        self.reduce = reduce
        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

    def _forward(self, data):
        z = data['atomic_numbers'].long()
        pos = data['positions']
        if self.regress_forces:
            pos.requires_grad_(True)

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors, self.offset_degree
            )
            pbc_distances = get_pbc_distances(
                data['positions'],
                edge_index,
                data['cell'],
                cell_offsets,
                neighbors,
                return_distances_vec=True
            )
            edge_weight = pbc_distances['distances']
            edge_index = pbc_distances['edge_index']

            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            data['batch'] = torch.zeros_like(z) if data['batch'] is None else data['batch']
            energy = scatter(h, data['batch'], dim=0, reduce=self.reduce)

        else:
            energy = super(SchNetWrap, self).forward(z, pos, data['batch'])

        return energy

    def forward(self, data):
        if self.regress_forces:
            data['positions'].requires_grad_(True)
        energy = self._forward(data)
        out = {}

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data['positions'],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            out['energy'] = energy
            out['forces'] = forces
            return out
        else:
            out['energy'] = energy
            return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())