"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

stpayu/grannfield: https://github.com/stpayu/grannfield

---

This code has been modified from the original version at
Open-Catalyst-Project/ocp: https://github.com/Open-Catalyst-Project/ocp
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from .embeddings import KHOT_EMBEDDINGS
from grannfield.utils.distances import GaussianSmearing
from grannfield.utils.utils import radius_graph_pbc, get_pbc_distances


class CGCNN(nn.Module):
    def __init__(self,
                 num_features: int = 64,
                 num_gaussians: int = 100,
                 use_pbc: bool = True,
                 regress_forces: bool = True,
                 num_targets: int = 1,
                 cutoff: float = 6.0,
                 max_neighbors_num: int = 25,
                 num_graph_conv_layers: int = 6,
                 fc_feat_size: int = 128,
                 num_fc_layers: int = 4,
                 offset_degree: int = 1,
                 ):
        super(CGCNN, self).__init__()
        self.num_features = num_features
        self.num_gaussians = num_gaussians
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_neighbors_num = max_neighbors_num
        self.fc_feat_size = fc_feat_size
        self.num_fc_layers = num_fc_layers
        self.offset_degree = offset_degree
        
        embeddings = KHOT_EMBEDDINGS
        
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.nuclear_embedding = nn.Linear(len(embeddings[1]), self.num_features)
        
        self.convs = nn.ModuleList(
            [
                Conv(
                    node_dim=self.num_features,
                    edge_dim=self.num_gaussians,
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_fc = nn.Sequential(
            nn.Linear(self.num_features, self.fc_feat_size), nn.Softplus()
        )
        
        if num_fc_layers > 1:
            layers = []
            for _ in range(num_fc_layers - 1):
                layers.append(nn.Linear(self.fc_feat_size, self.fc_feat_size))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(self.fc_feat_size, self.num_targets)

        self.expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        
    def _forward(self, data):
        # Get node features
        if self.embedding.device != data['atomic_numbers'].device:
            self.embedding = self.embedding.to(data['atomic_numbers'].device)
        data['x'] = self.embedding[data['atomic_numbers'].long() - 1]

        if self.use_pbc:
            neighbors_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors_num, self.offset_degree
            )
            pbc_distances = get_pbc_distances(
                data['positions'],
                neighbors_index,
                data['cell'],
                cell_offsets,
                neighbors,
                return_distances_vec=True
            )
            distances = pbc_distances['distances']
            data['neighbors_index'] = pbc_distances['edge_index']
        else:
            neighbors_index = radius_graph(
                data['positions'], r=self.cutoff
            )
            row, col = neighbors_index
            data['neighbors_index'] = neighbors_index
            # neighbors = compute_neighbors(data, neighbors_index)
            distances = (data['positions'][row] - data['positions'][col]).norm(dim=-1)

        distance_expansion = self.expansion(distances)

        data['neighbors_feats'] = distance_expansion

        # Forward pass through the network
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, 'fcs'):
            mol_feats = self.fcs(mol_feats)

        y = self.fc_out(mol_feats)
        return y

    def forward(self, data):
        if self.regress_forces:
            data['positions'].requires_grad_(True)
        y = self._forward(data)
        out = {}
        out['energy'] = y

        if self.regress_forces:
            dy = -1 * (
                torch.autograd.grad(
                    y,
                    data['positions'],
                    grad_outputs=torch.ones_like(y),
                    create_graph=True,
                )[0]
            )
            out['forces'] = dy

        return out

    def _convolve(self, data):
        """
        Returns the output of the convolution layers before they are passed
        into the dense layers.
        """
        node_feats = self.nuclear_embedding(data['x'])
        for f in self.convs:
            node_feats = f(node_feats, data['neighbors_index'], data['neighbors_feats'])

        mol_feats = scatter(node_feats, data['batch'], dim=0, reduce='mean')

        return mol_feats
        
        
class Conv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim, max_neighbors_radius=6.0, **kwargs):
        super(Conv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.max_neighbors_radius = max_neighbors_radius

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size,
            2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)

        self.lin1.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, neighbors_index, neighbors_feats):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            neighbors_index has shape [2, num_edges]
            neighbors_feats is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            neighbors_index, x=x, neighbors_feats=neighbors_feats, size=(x.size(0), x.size(0))
        )
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, neighbors_feats):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            neighbors_feats has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, neighbors_feats], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2