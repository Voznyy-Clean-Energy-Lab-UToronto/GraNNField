"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

stpayu/grannfield: https://github.com/stpayu/grannfield

---

This code has been modified from the original version at
Open-Catalyst-Project/ocp: https://github.com/Open-Catalyst-Project/ocp
"""

import torch
from torch import nn
from torch_geometric.nn import DimeNet, radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from grannfield.utils.utils import (
    get_pbc_distances,
    radius_graph_pbc,
    compute_neighbors
)


class DimeNetWrap(DimeNet):
    r"""Wrapper around the directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.

    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_blocks (int, optional): Number of building blocks.
            (default: :obj:`6`)
        num_bilinear (int, optional): Size of the bilinear layer tensor.
            (default: :obj:`8`)
        num_spherical (int, optional): Number of spherical harmonics.
            (default: :obj:`7`)
        num_radial (int, optional): Number of radial basis functions.
            (default: :obj:`6`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        max_angles_per_image (int, optional): The maximum number of angles used
            per image. This can be used to reduce memory usage at the cost of
            model performance. (default: :obj:`1e6`)
    """

    def __init__(
        self,
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        max_angles_per_image=int(1e6),
        offset_degree=1
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.max_angles_per_image = max_angles_per_image
        self.max_neighbors = 50
        self.offset_degree = offset_degree

        super(DimeNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def _forward(self, data):
        pos = data['positions']
        if self.regress_forces:
            pos.requires_grad_(True)

        if self.use_pbc:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors, self.offset_degree
            )
            pbc_distances = get_pbc_distances(
                data['positions'],
                edge_index,
                data['cell'],
                cell_offsets,
                neighbors,
                return_distances_vec=False,
                return_offsets=True,
            )
            offsets = pbc_distances['offsets']
            data['cell_offsets'] = cell_offsets
            dist = pbc_distances['distances']
            edge_index = pbc_distances['edge_index']
        else:
            edge_index = radius_graph(
                data['positions'], r=self.cutoff
            )
            row, col = edge_index
            dist = (data['positions'][row] - data['positions'][col]).norm(dim=-1)

        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data['cell_offsets'],
            num_nodes=data['atomic_numbers'].size(0),
        )

        # Cap no. of triplets during training.
        if self.training:
            sub_ix = torch.randperm(idx_i.size(0))[
                : self.max_angles_per_image * data['atomic_numbers'].size(0)
            ]
            idx_i, idx_j, idx_k = (
                idx_i[sub_ix],
                idx_j[sub_ix],
                idx_k[sub_ix],
            )
            idx_kj, idx_ji = idx_kj[sub_ix], idx_ji[sub_ix]

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(data['atomic_numbers'].long(), rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.size(0))

        energy = P.sum(dim=0) if data['batch'] is None else scatter(P, data['batch'], dim=0)

        return energy

    def forward(self, data):
        if self.regress_forces:
            data['positions'].requires_grad_(True)
        out = {}
        energy = self._forward(data)
        out['energy'] = energy

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data['positions'],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            out['forces'] = forces
            return out
        else:
            return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())