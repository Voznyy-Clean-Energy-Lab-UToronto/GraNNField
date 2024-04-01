"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

stpayu/grannfield: https://github.com/stpayu/grannfield

---

This code has been modified from the original version at
shehzaidi/pre-training-via-denoising: https://github.com/shehzaidi/pre-training-via-denoising

"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import radius_graph, MessagePassing
from torch_scatter import scatter

from grannfield.utils.distances import GaussianSmearing, CosineCutoff
from grannfield.utils.utils import radius_graph_pbc, get_pbc_distances


class GNS_TAT(nn.Module):
    def __init__(self,
                 num_features: int = 64,
                 num_gaussians: int = 100,
                 use_pbc: bool = True,
                 regress_forces: bool = False,
                 num_targets: int = 1,
                 cutoff: float = 6.0,
                 max_neighbors_num: int = 50,
                 num_attn_layers: int = 6,
                 max_z: int = 100,
                 intm_activation: Optional = nn.SiLU(),
                 attn_activation_1: Optional = nn.SiLU(),
                 attn_activation_2: Optional = nn.SiLU(),
                 offset_degree: int = 1
                 ):
        super(GNS_TAT, self).__init__()
        self.num_features = num_features
        self.num_gaussians = num_gaussians
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_neighbors_num = max_neighbors_num
        self.attn_activation_1 = attn_activation_1
        self.attn_activation_2 = attn_activation_2
        self.intm_activation= intm_activation
        self.max_z = max_z
        self.offset_degree = offset_degree

        self.embedding = nn.Embedding(self.max_z, num_features)

        # Vector
        self.attns = nn.ModuleList(
            [
                EquivariantMultiHeadAttention(
                    num_features=self.num_features,
                    num_rbf=self.num_gaussians,
                    use_dk_proj=True,
                    use_dv_proj=True,
                    num_heads=8,
                    attn_activation_1=self.attn_activation_1,
                    attn_activation_2=self.attn_activation_2,
                    cutoff_lower=0,
                    cutoff_upper=self.cutoff
                    )
                for _ in range(num_attn_layers)
            ]
        )

        self.lyn = nn.LayerNorm(self.num_features)
        self.lyn_vec = EquivariantLayerNorm(self.num_features)

        self.intms = Intm(self.num_features, self.intm_activation, self.num_targets)

        self.expansion = GaussianSmearing(0, self.cutoff, self.num_gaussians)

    def reset_parameters(self):
        pass

    def _forward(self, data):
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
            distances_vec = pbc_distances['distances_vec']
            neighbors_index = pbc_distances['edge_index']
        else:
            neighbors_index = radius_graph(
                data['positions'], r=self.cutoff
            )
            row, col = neighbors_index
            # neighbors = compute_neighbors(data, neighbors_index)
            distances = (data['positions'][row] - data['positions'][col]).norm(dim=-1)
            distances_vec = data['positions'][row] - data['positions'][col]

        distance_expansion = self.expansion(distances)
        mask = neighbors_index[0] != neighbors_index[1]
        distances_vec[mask] = distances_vec[mask] / torch.norm(distances_vec[mask], dim=1).unsqueeze(1)

        x = self.embedding(data['atomic_numbers'])

        vec = torch.zeros(x.size(0), 3, self.num_features, device=x.device)

        for attn in self.attns:
            dx, dvec = attn(x, vec, neighbors_index, distances, distance_expansion, distances_vec)
            x = x + dx
            vec = vec + dvec

        x = self.lyn(x)
        v = self.lyn_vec(vec)

        x = self.intms(x, v)

        x = scatter(x, data['batch'], dim=0, reduce='mean')

        return x

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


class Intm(nn.Module):
    def __init__(self, num_features, intm_activation, num_targets):
        super(Intm, self).__init__()
        self.intms_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    num_features,
                    num_features // 2,
                    intm_activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    num_features // 2,
                    num_targets,
                    intm_activation
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.intms_network:
            layer.reset_parameters()

    def forward(self, x, v):
        for layer in self.intms_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        num_features,
        num_rbf,
        use_dk_proj,
        use_dv_proj,
        num_heads,
        attn_activation_1,
        attn_activation_2,
        cutoff_lower,
        cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr='add', node_dim=0)
        assert num_features % num_heads == 0, (
            f"The number of hidden channels ({num_features}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.use_dk_proj = use_dk_proj
        self.use_dv_proj = use_dv_proj
        self.num_heads = num_heads
        self.hidden_channels = num_features
        self.head_dim = num_features // num_heads

        self.layernorm = nn.LayerNorm(num_features)
        self.attn_activation_1 = attn_activation_1
        self.attn_activation_2 = attn_activation_2
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(num_features, num_features)
        self.k_proj = nn.Linear(num_features, num_features)
        self.v_proj = nn.Linear(num_features, num_features * 3)
        self.o_proj = nn.Linear(num_features, num_features * 3)

        self.vec_proj = nn.Linear(num_features, num_features * 3)

        self.dk_proj = None

        if self.use_dk_proj:
            self.dk_proj = nn.Linear(num_rbf, num_features)

        self.dv_proj = None

        if self.use_dv_proj:
            self.dv_proj = nn.Linear(num_rbf, num_features * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        self.vec_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.attn_activation_1(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.attn_activation_1(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation_2(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter('weight', None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        activation,
        intermediate_channels=None,
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels)

        self.activation = activation
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            self.activation,
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = self.activation if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        self.vec1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        self.vec1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v