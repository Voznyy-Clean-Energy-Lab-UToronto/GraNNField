"""
Implementation of MEGNet class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, MessagePassing, Set2Set
from torch_scatter import scatter

from grannfield.models.megnet.layers import GraphNetwork as GN
from grannfield.utils.distances import GaussianSmearing
from grannfield.utils.utils import radius_graph_pbc, get_pbc_distances

class GraphNetwork(nn.Module):
    """Graph Networks

    A Graph Network [1]_ takes a graph as input and returns an updated graph
    as output. The output graph has same structure as input graph but it
    has updated node features, edge features and global state features.

    Parameters
    ----------
    n_node_features: int
        Number of features in a node
    n_edge_features: int
        Number of features in a edge
    n_global_features: int
        Number of global features
    is_undirected: bool, optional (default True)
        Directed or undirected graph
    residual_connection: bool, optional (default True)
        If True, the layer uses a residual connection during training

    Example
    -------
    # >>> import torch
    # >>> from deepchem.models.torch_models.layers import GraphNetwork as GN
    # >>> n_nodes, n_node_features = 5, 10
    # >>> n_edges, n_edge_features = 5, 2
    # >>> n_global_features = 4
    # >>> node_features = torch.randn(n_nodes, n_node_features)
    # >>> edge_features = torch.randn(n_edges, n_edge_features)
    # >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
    # >>> global_features = torch.randn(1, n_global_features)
    # >>> gn = GN(n_node_features=n_node_features, n_edge_features=n_edge_features, n_global_features=n_global_features)
    # >>> node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)

    References
    ----------
    .. [1] Battaglia et al, Relational inductive biases, deep learning, and graph networks. https://arxiv.org/abs/1806.01261 (2018)
  """

    def __init__(self,
                 n_node_features: int = 32,
                 n_edge_features: int = 32,
                 n_global_features: int = 32,
                 is_undirected: bool = True,
                 residual_connection: bool = True):
        super().__init__()
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_global_features = n_global_features
        self.is_undirected = is_undirected
        self.residual_connection = residual_connection

        self.edge_models, self.node_models, self.global_models = torch.nn.ModuleList(
        ), torch.nn.ModuleList(), torch.nn.ModuleList()
        self.edge_models.append(
            nn.Linear(in_features=n_node_features * 2 + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.node_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.global_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))

        # Used for converting edges back to their original shape
        self.edge_dense = nn.Linear(in_features=32,
                                    out_features=n_edge_features)
        self.node_dense = nn.Linear(in_features=32,
                                    out_features=n_node_features)
        self.global_dense = nn.Linear(in_features=32,
                                      out_features=n_global_features)

    def reset_parameters(self) -> None:
        self.edge_dense.reset_parameters()
        self.node_dense.reset_parameters()
        self.global_dense.reset_parameters()
        for i in range(0, len(self.edge_models)):
            self.edge_models[i].reset_parameters()
        for i in range(0, len(self.node_models)):
            self.node_models[i].reset_parameters()
        for i in range(0, len(self.global_models)):
            self.global_models[i].reset_parameters()

    def _update_edge_features(self, node_features, edge_index, edge_features,
                              global_features, batch):
        src_index, dst_index = edge_index
        out = torch.cat((node_features[src_index], node_features[dst_index],
                         edge_features, global_features[batch]),
                        dim=1)
        assert out.shape[
            1] == self.n_node_features * 2 + self.n_edge_features + self.n_global_features
        for model in self.edge_models:
            out = model(out)
        return self.edge_dense(out)

    def _update_node_features(self, node_features, edge_index, edge_features,
                              global_features, batch):

        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter(src=edge_features,
                                             index=dst_index,
                                             dim=0,
                                             reduce='mean')
        out = torch.cat(
            (node_features, edge_features_mean_by_node, global_features[batch]),
            dim=1)
        for model in self.node_models:
            out = model(out)
        return self.node_dense(out)

    def _update_global_features(self, node_features, edge_features,
                                global_features, node_batch_map,
                                edge_batch_map):
        edge_features_mean = scatter(src=edge_features,
                                     index=edge_batch_map,
                                     dim=0,
                                     reduce='mean')
        node_features_mean = scatter(src=node_features,
                                     index=node_batch_map,
                                     dim=0,
                                     reduce='mean')
        out = torch.cat(
            (edge_features_mean, node_features_mean, global_features), dim=1)
        for model in self.global_models:
            out = model(out)
        return self.global_dense(out)

    def forward(
            self,
            node_features,
            edge_index,
            edge_features,
            global_features,
            batch = None):
        """Output computation for a GraphNetwork

        Parameters
        ----------
        node_features: torch.Tensor
            Input node features of shape :math:`(|\mathcal{V}|, F_n)`
        edge_index: torch.Tensor
            Edge indexes of shape :math:`(2, |\mathcal{E}|)`
        edge_features: torch.Tensor
            Edge features of the graph, shape: :math:`(|\mathcal{E}|, F_e)`
        global_features: torch.Tensor
            Global features of the graph, shape: :math:`(F_g, 1)` where, :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the number of nodes and edges in the graph,
            :math:`F_n`, :math:`F_e`, :math:`F_g` denotes the number of node features, edge features and global state features respectively.
        batch: torch.LongTensor (optional, default: None)
            A vector that maps each node to its respective graph identifier. The attribute is used only when more than one graph are batched together during a single forward pass.
        """
        if batch is None:
            batch = node_features.new_zeros(node_features.size(0),
                                            dtype=torch.int64)

        node_features_copy, edge_features_copy, global_features_copy = node_features, edge_features, global_features
        if self.is_undirected is True:
            # holding bi-directional edges in case of undirected graphs
            edge_index = torch.cat((edge_index, edge_index.flip([0])), dim=1)
            edge_features_len = edge_features.shape[0]
            edge_features = torch.cat((edge_features, edge_features), dim=0)
        edge_batch_map = batch[edge_index[0]]
        edge_features = self._update_edge_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features,
                                                   edge_batch_map)
        node_features = self._update_node_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features, batch)
        global_features = self._update_global_features(node_features,
                                                       edge_features,
                                                       global_features, batch,
                                                       edge_batch_map)

        if self.is_undirected is True:
            # coonverting edge features to its original shape
            split = torch.split(edge_features,
                                [edge_features_len, edge_features_len])
            edge_features = (split[0] + split[1]) / 2

        if self.residual_connection:
            edge_features += edge_features_copy
            node_features += node_features_copy
            global_features += global_features_copy

        return node_features, edge_features, global_features

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(n_node_features={self.n_node_features}, n_edge_features={self.n_edge_features}, n_global_features={self.n_global_features}, is_undirected={self.is_undirected}, residual_connection={self.residual_connection})'
        )

class MEGNet(nn.Module):
    """MatErials Graph Network

    A model for predicting crystal and molecular properties using GraphNetworks.

    Example
    -------
    # >>> import numpy as np
    # >>> from torch_geometric.data import Batch
    # >>> from deepchem.feat import GraphData
    # >>> n_nodes, n_node_features = 5, 10
    # >>> n_edges, n_edge_attrs = 5, 2
    # >>> n_global_features = 4
    # >>> node_features = np.random.randn(n_nodes, n_node_features)
    # >>> edge_attrs = np.random.randn(n_edges, n_edge_attrs)
    # >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    # >>> global_features = np.random.randn(1, n_global_features)
    # >>> graph = GraphData(node_features, edge_index, edge_attrs, global_features=global_features)
    # >>> batch = Batch()
    # >>> batch = batch.from_data_list([graph.to_pyg_graph()])
    # >>> model = MEGNet(n_node_features=n_node_features, n_edge_features=n_edge_attrs, n_global_features=n_global_features)
    # >>> pred = model(batch)

    Note
    ----
    This class requires torch-geometric to be installed.
    """

    def __init__(self,
                 n_node_features: int = 32,
                 n_edge_features: int = 32,
                 n_global_features: int = 32,
                 n_blocks: int = 1,
                 use_pbc = True,
                 regress_forces = True,
                 max_neighbors_num = 12,
                 cutoff = 6.0,
                 is_undirected: bool = True,
                 residual_connection: bool = True,
                 n_classes: int = 2,
                 num_targets: int = 1,
                 offset_degree: int = 1):
        """

        Parameters
        ----------
        n_node_features: int
            Number of features in a node
        n_edge_features: int
            Number of features in a edge
        n_global_features: int
            Number of global features
        n_blocks: int
            Number of GraphNetworks block to use in update
        is_undirected: bool, optional (default True)
            True when the graph is undirected graph , otherwise False
        residual_connection: bool, optional (default True)
            If True, the layer uses a residual connection during training
        n_tasks: int, default 1
            The number of tasks
        mode: str, default 'regression'
            The model type - classification or regression
        n_classes: int, default 2
            The number of classes to predict (used only in classification mode).
        """
        super(MEGNet, self).__init__()

        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_global_features = n_global_features
        self.megnet_blocks = nn.ModuleList()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            self.megnet_blocks.append(
                GN(n_node_features=n_node_features,
                   n_edge_features=n_edge_features,
                   n_global_features=n_global_features,
                   is_undirected=is_undirected,
                   residual_connection=residual_connection))
        self.n_tasks = num_targets
        self.n_classes = n_classes
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.max_neighbors_num = max_neighbors_num
        self.cutoff = cutoff
        self.offset_degree = offset_degree

        self.max_z = 100
        self.embedding = nn.Embedding(self.max_z, n_node_features)

        self.expansion = GaussianSmearing(0, self.cutoff, self.n_edge_features)

        self.set2set_nodes = Set2Set(in_channels=n_node_features,
                                     processing_steps=3,
                                     num_layers=1)
        self.set2set_edges = Set2Set(in_channels=n_edge_features,
                                     processing_steps=3,
                                     num_layers=1)

        self.dense = nn.Sequential(
            nn.Linear(in_features=2 * n_node_features + 2 * n_edge_features +
                      n_global_features,
                      out_features=32),
            nn.Linear(in_features=32, out_features=16))

        self.out = nn.Linear(in_features=16, out_features=num_targets)

    def _forward(self, data):
        if self.use_pbc:
            data['edge_index'], cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors_num, self.offset_degree
            )
            pbc_distances = get_pbc_distances(
                data['positions'],
                data['edge_index'],
                data['cell'],
                cell_offsets,
                neighbors,
                return_distances_vec=True
            )
            data['edge_index'] = pbc_distances['edge_index']
            distances = pbc_distances['distances']
        else:
            data['edge_index'] = radius_graph(
                data['positions'], r=self.cutoff
            )
            row, col = data['edge_index']
            distances = (data['positions'][row] - data['positions'][col]).norm(dim=-1)

        node_features = self.embedding(data['atomic_numbers'])
        data['edge_attr'] = self.expansion(distances)
        edge_index, edge_features = data['edge_index'], data['edge_attr']
        global_features = data['global_features']
        batch = data['batch']

        for i in range(self.n_blocks):
            node_features, edge_features, global_features = self.megnet_blocks[
                i](node_features, edge_index, edge_features, global_features,
                   batch)

        node_features = self.set2set_nodes(node_features, batch)
        edge_features = self.set2set_edges(edge_features, batch[edge_index[0]])
        y = torch.cat([node_features, edge_features, global_features], axis=1)
        y = self.out(self.dense(y))

        return y

    def forward(self, data):
        if self.regress_forces:
            data['positions'].requires_grad = True
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
