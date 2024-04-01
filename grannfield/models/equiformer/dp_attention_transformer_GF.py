import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3

import torch_geometric
import math

from grannfield.utils.utils import radius_graph_pbc, get_pbc_distances

from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, DropPath
from .graph_attention_transformer import (get_norm_layer, 
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct, SeparableFCTP,
    Vec2AttnHeads, AttnHeads2Vec,
    GraphAttention, FeedForwardNetwork, 
    TransBlock, 
    NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, ScaledScatter
)
from .dp_attention_transformer import DotProductAttention, DPTransBlock
from .gaussian_rbf import GaussianRadialBasisLayer

_RESCALE = True
_USE_BIAS = True

# OC20
_MAX_ATOM_TYPE = 84

# Statistics of LPGS
_AVG_NUM_NODES = 50.0
_AVG_DEGREE = 44.46


class DotProductAttentionTransformerGF(torch.nn.Module):
    '''
        Differences from GraphAttentionTransformer:
            1. Use `otf_graph` and `use_pbc`. `otf_graph` corresponds to whether to 
                build edges on the fly for each inputs. `use_pbc` corresponds to whether
                to consider periodic boundary condition.
            2. Use OC20 registry.
            3. Use `max_neighbors` following models in OC20.
            4. The first two input arguments (e.g., num_atoms and bond_feat_dim) are 
                not used. They are there because of trainer takes extra arguments.
    ''' 
    def __init__(self,
                 irreps_node_embedding='256x0e+128x1e',
                 num_layers=6,
                 irreps_node_attr='1x0e',
                 use_node_attr=False,
                 irreps_sh='1x0e+1x1e',
                 max_radius=6.0,
                 number_of_basis=128,
                 fc_neurons=[64, 64],
                 use_atom_edge_attr=False,
                 irreps_atom_edge_attr='8x0e',
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1e',
                 num_heads=8,
                 irreps_pre_attn=None,
                 rescale_degree=False,
                 nonlinear_message=False,
                 irreps_mlp_mid='768x0e+384x1e',
                 norm_layer='layer',
                 alpha_drop=0.2,
                 proj_drop=0.0,
                 out_drop=0.0,
                 drop_path_rate=0.0,
                 use_auxiliary_task=False,
                 use_pbc=True,
                 regress_forces=True,
                 max_neighbors=50,
                 offset_degree=1):
        
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        self.regress_forces= regress_forces
        self.use_pbc = use_pbc
        self.max_neighbors = max_neighbors
        self.offset_degree = offset_degree
        
        self.use_node_attr = use_node_attr
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        #if not self.use_node_attr:
        #    assert self.irreps_node_attr == o3.Irreps('1x0e')
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.use_atom_edge_attr = use_atom_edge_attr
        self.irreps_atom_edge_attr = o3.Irreps(irreps_atom_edge_attr)
        temp = 0
        if self.use_atom_edge_attr:
            for _, ir in self.irreps_atom_edge_attr:
                assert ir.is_scalar()
            temp = 2 * self.irreps_atom_edge_attr.dim
        self.fc_neurons = [temp + self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        
        self.attr_embed = None
        if self.use_node_attr:
            self.attr_embed = NodeEmbeddingNetwork(self.irreps_node_attr, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.edge_src_embed = None
        self.edge_dst_embed = None
        if self.use_atom_edge_attr:
            self.edge_src_embed = NodeEmbeddingNetwork(self.irreps_atom_edge_attr, _MAX_ATOM_TYPE)
            self.edge_dst_embed = NodeEmbeddingNetwork(self.irreps_atom_edge_attr, _MAX_ATOM_TYPE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantScalarsDropout(self.irreps_feature, self.out_drop)
        
        self.irreps_feature_scalars = []
        for mul, ir in self.irreps_feature:
            if ir.l == 0 and ir.p == 1:
                self.irreps_feature_scalars.append((mul, (ir.l, ir.p)))
        self.irreps_feature_scalars = o3.Irreps(self.irreps_feature_scalars)
        
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature_scalars, rescale=_RESCALE), 
            Activation(self.irreps_feature_scalars, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature_scalars, o3.Irreps('1x0e')))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.use_auxiliary_task = use_auxiliary_task
        if self.use_auxiliary_task:
            irreps_out_auxiliary = o3.Irreps('1x1o')
            if o3.Irrep('1o') not in self.irreps_feature:
                irreps_out_auxiliary = o3.Irreps('1x1e')
            self.auxiliary_head = DotProductAttention(self.irreps_feature, 
                self.irreps_node_attr, self.irreps_edge_attr, irreps_out_auxiliary,
                self.fc_neurons, 
                self.irreps_head, self.num_heads, self.irreps_pre_attn, 
                self.rescale_degree, 
                self.alpha_drop, proj_drop=0.0)
            
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = DPTransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            

    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        
        return set(no_wd_list)

    def _forward(self, data):
        if self.use_pbc:
            edge_index, offsets, _, neighbors = radius_graph_pbc(
                data, self.max_radius, self.max_neighbors, self.offset_degree
            )
            pbc_distances = get_pbc_distances(
                data['positions'],
                edge_index,
                data['cell'],
                offsets,
                neighbors,
                return_distances_vec=True
            )
            edge_length = pbc_distances['distances']
            edge_vec = pbc_distances['distances_vec']
            edge_index = pbc_distances['edge_index']
        else:
            edge_index = radius_graph(
                data['positions'], r=self.cutoff
            )
            row, col = edge_index
            edge_length = (data['positions'][row] - data['positions'][col]).norm(dim=-1)
            edge_vec = data['positions'][row] - data['positions'][col]

        batch = data['batch']
        
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        # Following Graphoformer, which encodes both atom type and tag
        atomic_numbers = data['atomic_numbers'].long()
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(atomic_numbers)
        
        edge_length_embedding = self.rbf(edge_length, atomic_numbers, 
            edge_src, edge_dst)
        if self.use_atom_edge_attr:
            src_attr, _, _ = self.edge_src_embed(atomic_numbers)
            dst_attr, _, _ = self.edge_dst_embed(atomic_numbers)
            edge_length_embedding = torch.cat((src_attr[edge_src], 
                dst_attr[edge_dst], edge_length_embedding), dim=1)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        
        if self.attr_embed is not None:
            node_attr, _, _ = self.attr_embed(atomic_numbers)
        else:
            node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
            
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        
        if self.out_dropout is not None:
            outputs = self.out_dropout(node_features)
        else:
            outputs = node_features
        outputs = self.head(outputs)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        # auxiliary IS2RS
        if self.use_auxiliary_task:
            outputs_aux = self.auxiliary_head(node_input=node_features, 
                node_attr=node_attr, edge_src=edge_src, edge_dst=edge_dst, 
                edge_attr=edge_sh, edge_scalars=edge_length_embedding, 
                batch=batch)
            return outputs, outputs_aux
        
        return outputs

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
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
