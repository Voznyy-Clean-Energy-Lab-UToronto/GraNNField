"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield
"""

import copy
import json
import logging
from argparse import Namespace

import numpy as np
import torch
from ase.db import connect
from base64 import b64decode

from torch_scatter import segment_coo, segment_csr
from torch_sparse import SparseTensor
from tqdm import tqdm

def set_random_seed(seed):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Args:
        seed (int, optional): if seed not present, it is generated based on time
    """
    import time
    import numpy as np

    # 1) if seed not present, generate based on time
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
                ((seed & 0xFF000000) >> 24)
                + ((seed & 0x00FF0000) >> 8)
                + ((seed & 0x0000FF00) << 8)
                + ((seed & 0x000000FF) << 24)
        )
    # 2) Set seed for numpy (e.g. splitting)
    np.random.seed(seed)
    # 3) Set seed for torch (manual_seed now seeds all CUDA devices automatically)
    torch.manual_seed(seed)
    logging.info("Random state initialized with seed {:<10d}".format(seed))


def count_params(model):
    """
    This function takes a modules as an input and returns the number of
    trainable parameters.

    Args:collect
        modules (AtomisticModel): modules for which you want to count
                                the trainable parameters

    Returns:
        params (int): number of trainable parameters for the modules
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def to_json(jsonpath, argparse_dict):
    """
    This function creates a .json file as a copy of argparse_dict

    Args:
        jsonpath (str): path to the .json file
        argparse_dict (dict): dictionary containing arguments from argument parser
    """
    with open(jsonpath, "w") as fp:
        json.dump(argparse_dict, fp, sort_keys=True, indent=4)


def read_from_json(jsonpath):
    """
    This function reads args from a .json file and returns the content as a namespace dict

    Args:
        jsonpath (str): path to the .json file

    Returns:
        namespace_dict (Namespace): namespace object build from the dict stored into the given .json file.
    """
    with open(jsonpath) as handle:
        dict = json.loads(handle.read())
        namespace_dict = Namespace(**dict)
    return namespace_dict


def load_model(model_path, map_location=None):
    """
    Wrapper function for `for safely loading models where certain new attributes of the modules class are not present.
    E.g. "requires_stress" for computing the stress tensor.

    Args:
        model_path (str): Path to the modules file.
        map_location (torch.device): Device where the modules should be loaded to.

    Returns:
        :class:`schnetpack.atomistic.AtomisticModel`: Loaded SchNetPack modules.

    """
    model = torch.load(model_path, map_location=map_location)

    # Check for data parallel models
    if hasattr(model, "modules"):
        model_module = model.module
        output_modules = model.module.output_modules
    else:
        model_module = model
        output_modules = model.output_modules

    # Set stress tensor attribute if not present
    if not hasattr(model_module, "requires_stress"):
        model_module.requires_stress = False
        for module in output_modules:
            module.stress = None

    return model


def read_deprecated_database(db_path):
    """
    Read all atoms and properties from deprecated ase databases.

    Args:
        db_path (str): Path to deprecated database

    Returns:
        atoms (list): All atoms objects of the database.
        properties (list): All property dictionaries of the database.

    """
    with connect(db_path) as conn:
        db_size = conn.count()
    atoms = []
    properties = []
    key_value_pairs = []

    for idx in tqdm(range(1, db_size + 1), "Reading deprecated database"):
        with connect(db_path) as conn:
            row = conn.get(idx)

        at = row.toatoms()
        pnames = [pname for pname in row.data.keys() if not pname.startswith("_")]
        props = {}
        for pname in pnames:
            try:
                shape = row.data['_shape_' + pname]
                dtype = row.data['_dtype_' + pname]
                prop = np.frombuffer(b64decode(row.data[pname]), dtype=dtype)
                prop = prop.reshape(shape)
            except:
                # fallback for properties stored directly
                # in the row
                if pname in row:
                    prop = row[pname]
                else:
                    prop = row.data[pname]

                try:
                    prop.shape
                except AttributeError as e:
                    prop = np.array([prop], dtype=np.float32)
            props[pname] = prop

        atoms.append(at)
        properties.append(props)
        key_value_pairs.append(row.key_value_pairs)

    return atoms, properties, key_value_pairs


def activate_stress_computation(model, stress="stress"):
    """
    Utility function to activate the computation of the stress tensor for a modules not trained explicitly on
    this property. It is recommended to at least have used forces during training when switching on the stress.
    Moreover, now proper crystal cell (volume > 0) needs to be specified for the molecules.

    Args:
        model (schnetpack.atomistic.AtomisticModel): SchNetPack modules for which computation of the stress tensor
                                                    should be activated.
        stress (str): Designated name of the stress tensor property used in the modules output.
    """
    # Check for data parallel models
    if hasattr(model, "modules"):
        model_module = model.module
        output_modules = model.module.output_modules
    else:
        model_module = model
        output_modules = model.output_modules

    # Set stress tensor attribute if not present
    if hasattr(model_module, 'requires_stress'):
        model_module.requires_stress = True
        for module in output_modules:
            if hasattr(module, 'stress'):
                module.stress = stress


def cut_off_function(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Cutoff function that smoothly goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff. For x >= cutoff, f(x) = 0. This function has
    infinitely many smooth derivatives. Only positive x should be used as input.
    """
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
    return torch.where(
        x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
    )


def radius_graph_pbc_oc(data, radius, max_num_neighbors_threshold):
    device = data['positions'].device
    batch_size = len(data['n_atoms'])

    # position of the atoms
    atom_pos = data['positions']

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data['n_atoms']
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
            torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (
                 torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode='trunc')
             ) + index_offset_expand
    index2 = (
                     atom_count_sqr % num_atoms_per_image_expand
             ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(data['cell'][:, 1], data['cell'][:, 2], dim=-1)
    cell_vol = torch.sum(data['cell'][:, 0] * cross_a2a3, dim=-1, keepdim=True)

    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    rep_a1 = torch.ceil(radius * inv_min_dist_a1)

    cross_a3a1 = torch.cross(data['cell'][:, 2], data['cell'][:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    rep_a2 = torch.ceil(radius * inv_min_dist_a2)

    cross_a1a2 = torch.cross(data['cell'][:, 0], data['cell'][:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    rep_a3 = torch.ceil(radius * inv_min_dist_a3)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cat(torch.meshgrid(cells_per_dim, indexing='ij'), dim=-1).reshape(-1, 3)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data['cell'], 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data['n_atoms'],
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    neighbors_index = torch.stack((index2, index1))

    return neighbors_index, unit_cell, num_neighbors_image


def radius_graph_pbc(data, radius, max_num_neighbors_threshold, offset_degree):
    OFFSET_LIST = [[i, j, k] for i in range(offset_degree*-1, offset_degree+1) for j in range(offset_degree*-1, offset_degree+1) for k in range(offset_degree*-1, offset_degree+1)]

    device = data['positions'].device
    batch_size = len(data['n_atoms'])

    # position of the atoms
    atom_pos = data['positions']

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data['n_atoms']
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
            torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
                 torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode='trunc')
             ) + index_offset_expand
    index2 = (
                     atom_count_sqr % num_atoms_per_image_expand
             ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data['cell'], 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    num_neighbors = torch.zeros(len(atom_pos), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
        ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(atom_pos) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms_per_image.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms_per_image, dim=0)
    num_neighbors_image = (
            _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    # return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image

    # If max_num_neighbors is below the threshold, return early
    if (
            max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0
    ):
        return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    # atom_distance_sqr.sqrt() distance

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(atom_pos) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index1 * max_num_neighbors
            + torch.arange(len(index1), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(atom_pos), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    neighbors_index = torch.stack((index2, index1))

    return neighbors_index, unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image


def get_max_neighbors_mask(
        natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
            max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index * max_num_neighbors
            + torch.arange(len(index), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def get_pbc_distances(
        pos,
        edge_index,
        cell,
        cell_offsets,
        neighbors,
        return_offsets=False,
        return_distances_vec=False,
):
    row, col = edge_index

    distances_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distances_vectors += offsets

    # compute distances
    distances = distances_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances)).to(cell.device)
    nonzero_idx = nonzero_idx[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        'edge_index': edge_index,
        'distances': distances,
    }

    if return_distances_vec:
        out['distances_vec'] = distances_vectors[nonzero_idx]

    if return_offsets:
        out['offsets'] = offsets[nonzero_idx]

    return out


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(
        ones, edge_index[1], dim_size=data['n_atoms'].sum()
    )

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data['n_atoms'].shape[0] + 1, device=data['positions'].device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data['n_atoms'], dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


def get_triplets(graph, num_atoms):
    """
    Get all input edges b->a for each output edge c->a.
    It is possible that b=c, as long as the edges are distinct
    (i.e. atoms b and c stem from different unit cells).

    Arguments
    ---------
    graph: dict of torch.Tensor
        Contains the graph's edge_index.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
        in: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        out: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
    """
    idx_s, idx_t = graph["edge_index"]  # c->a (source=c, target=a)
    num_edges = idx_s.size(0)

    value = torch.arange(num_edges, device=idx_s.device, dtype=idx_s.dtype)
    # Possibly contains multiple copies of the same edge (for periodic interactions)
    adj = SparseTensor(
        row=idx_t,
        col=idx_s,
        value=value,
        sparse_sizes=(num_atoms, num_atoms),
    )
    adj_edges = adj[idx_t]

    # Edge indices (b->a, c->a) for triplets.
    idx = {}
    idx["in"] = adj_edges.storage.value()
    idx["out"] = adj_edges.storage.row()

    # Remove self-loop triplets
    # Compare edge indices, not atom indices to correctly handle periodic interactions
    mask = idx["in"] != idx["out"]
    idx["in"] = idx["in"][mask]
    idx["out"] = idx["out"][mask]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_mixed_triplets(
        graph_in,
        graph_out,
        num_atoms,
        to_outedge=False,
        return_adj=False,
        return_agg_idx=False,
):
    """
    Get all output edges (ingoing or outgoing) for each incoming edge.
    It is possible that in atom=out atom, as long as the edges are distinct
    (i.e. they stem from different unit cells). In edges and out edges stem
    from separate graphs (hence "mixed") with shared atoms.

    Arguments
    ---------
    graph_in: dict of torch.Tensor
        Contains the input graph's edge_index and cell_offset.
    graph_out: dict of torch.Tensor
        Contains the output graph's edge_index and cell_offset.
        Input and output graphs use the same atoms, but different edges.
    num_atoms: int
        Total number of atoms.
    to_outedge: bool
        Whether to map the output to the atom's outgoing edges a->c
        instead of the ingoing edges c->a.
    return_adj: bool
        Whether to output the adjacency (incidence) matrix between output
        edges and atoms adj_edges.
    return_agg_idx: bool
        Whether to output the indices enumerating the intermediate edges
        of each output edge.

    Returns
    -------
    Dictionary containing the entries:
        in: torch.Tensor, shape (num_triplets,)
            Indices of input edges
        out: torch.Tensor, shape (num_triplets,)
            Indices of output edges
        adj_edges: SparseTensor, shape (num_edges, num_atoms)
            Adjacency (incidence) matrix between output edges and atoms,
            with values specifying the input edges.
            Only returned if return_adj is True.
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
            Only returned if return_agg_idx is True.
    """
    idx_out_s, idx_out_t = graph_out["edge_index"]
    # c->a (source=c, target=a)
    idx_in_s, idx_in_t = graph_in["edge_index"]
    num_edges = idx_out_s.size(0)

    value_in = torch.arange(
        idx_in_s.size(0), device=idx_in_s.device, dtype=idx_in_s.dtype
    )
    # This exploits that SparseTensor can have multiple copies of the same edge!
    adj_in = SparseTensor(
        row=idx_in_t,
        col=idx_in_s,
        value=value_in,
        sparse_sizes=(num_atoms, num_atoms),
    )
    if to_outedge:
        adj_edges = adj_in[idx_out_s]
    else:
        adj_edges = adj_in[idx_out_t]

    # Edge indices (b->a, c->a) for triplets.
    idx_in = adj_edges.storage.value()
    idx_out = adj_edges.storage.row()

    # Remove self-loop triplets c->a<-c or c<-a<-c
    # Check atom as well as cell offset
    if to_outedge:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_t[idx_out]
        cell_offsets_sum = (
                graph_out["cell_offset"][idx_out] + graph_in["cell_offset"][idx_in]
        )
    else:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_s[idx_out]
        cell_offsets_sum = (
                graph_out["cell_offset"][idx_out] - graph_in["cell_offset"][idx_in]
        )
    mask = (idx_atom_in != idx_atom_out) | torch.any(
        cell_offsets_sum != 0, dim=-1
    )

    idx = {}
    if return_adj:
        idx["adj_edges"] = masked_select_sparsetensor_flat(adj_edges, mask)
        idx["in"] = idx["adj_edges"].storage.value().clone()
        idx["out"] = idx["adj_edges"].storage.row()
    else:
        idx["in"] = idx_in[mask]
        idx["out"] = idx_out[mask]

    if return_agg_idx:
        # idx['out'] has to be sorted
        idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_quadruplets(
        main_graph,
        qint_graph,
        num_atoms,
):
    """
    Get all d->b for each edge c->a and connection b->a
    Careful about periodic images!
    Separate interaction cutoff not supported.

    Arguments
    ---------
    main_graph: dict of torch.Tensor
        Contains the main graph's edge_index and cell_offset.
        The main graph defines which edges are embedded.
    qint_graph: dict of torch.Tensor
        Contains the quadruplet interaction graph's edge_index and
        cell_offset. main_graph and qint_graph use the same atoms,
        but different edges.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
        triplet_in['in']: torch.Tensor, shape (nTriplets,)
            Indices of input edge d->b in triplet d->b->a.
        triplet_in['out']: torch.Tensor, shape (nTriplets,)
            Interaction indices of output edge b->a in triplet d->b->a.
        triplet_out['in']: torch.Tensor, shape (nTriplets,)
            Interaction indices of input edge b->a in triplet c->a<-b.
        triplet_out['out']: torch.Tensor, shape (nTriplets,)
            Indices of output edge c->a in triplet c->a<-b.
        out: torch.Tensor, shape (nQuadruplets,)
            Indices of output edge c->a in quadruplet
        trip_in_to_quad: torch.Tensor, shape (nQuadruplets,)
            Indices to map from input triplet d->b->a
            to quadruplet d->b->a<-c.
        trip_out_to_quad: torch.Tensor, shape (nQuadruplets,)
            Indices to map from output triplet c->a<-b
            to quadruplet d->b->a<-c.
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
    """
    idx_s, _ = main_graph["edge_index"]
    idx_qint_s, _ = qint_graph["edge_index"]
    # c->a (source=c, target=a)
    num_edges = idx_s.size(0)
    idx = {}

    idx["triplet_in"] = get_mixed_triplets(
        main_graph,
        qint_graph,
        num_atoms,
        to_outedge=True,
        return_adj=True,
    )
    # Input triplets d->b->a

    idx["triplet_out"] = get_mixed_triplets(
        qint_graph,
        main_graph,
        num_atoms,
        to_outedge=False,
    )
    # Output triplets c->a<-b

    # ---------------- Quadruplets -----------------
    # Repeat indices by counting the number of input triplets per
    # intermediate edge ba. segment_coo assumes sorted idx['triplet_in']['out']
    ones = (
        idx["triplet_in"]["out"]
        .new_ones(1)
        .expand_as(idx["triplet_in"]["out"])
    )
    num_trip_in_per_inter = segment_coo(
        ones, idx["triplet_in"]["out"], dim_size=idx_qint_s.size(0)
    )

    num_trip_out_per_inter = num_trip_in_per_inter[idx["triplet_out"]["in"]]
    idx["out"] = torch.repeat_interleave(
        idx["triplet_out"]["out"], num_trip_out_per_inter
    )
    idx_inter = torch.repeat_interleave(
        idx["triplet_out"]["in"], num_trip_out_per_inter
    )
    idx["trip_out_to_quad"] = torch.repeat_interleave(
        torch.arange(
            len(idx["triplet_out"]["out"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        num_trip_out_per_inter,
    )

    # Generate input indices by using the adjacency
    # matrix idx['triplet_in']['adj_edges']
    idx["triplet_in"]["adj_edges"].set_value_(
        torch.arange(
            len(idx["triplet_in"]["in"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        layout="coo",
    )
    adj_trip_in_per_trip_out = idx["triplet_in"]["adj_edges"][
        idx["triplet_out"]["in"]
    ]
    # Rows in adj_trip_in_per_trip_out are intermediate edges ba
    idx["trip_in_to_quad"] = adj_trip_in_per_trip_out.storage.value()
    idx_in = idx["triplet_in"]["in"][idx["trip_in_to_quad"]]

    # Remove quadruplets with c == d
    # Triplets should already ensure that a != d and b != c
    # Compare atom indices and cell offsets
    idx_atom_c = idx_s[idx["out"]]
    idx_atom_d = idx_s[idx_in]

    cell_offset_cd = (
            main_graph["cell_offset"][idx_in]
            + qint_graph["cell_offset"][idx_inter]
            - main_graph["cell_offset"][idx["out"]]
    )
    mask_cd = (idx_atom_c != idx_atom_d) | torch.any(
        cell_offset_cd != 0, dim=-1
    )

    idx["out"] = idx["out"][mask_cd]
    idx["trip_out_to_quad"] = idx["trip_out_to_quad"][mask_cd]
    idx["trip_in_to_quad"] = idx["trip_in_to_quad"][mask_cd]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res


def repeat_blocks(
        sizes,
        repeats,
        continuous_indexing=True,
        start_idx=0,
        block_inc=0,
        repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def masked_select_sparsetensor_flat(src, mask):
    row, col, value = src.coo()
    row = row[mask]
    col = col[mask]
    value = value[mask]
    return SparseTensor(
        row=row, col=col, value=value, sparse_sizes=src.sparse_sizes()
    )


def calculate_interatomic_vectors(R, id_s, id_t, offsets_st):
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Arguments
    ---------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    # ReLU prevents negative numbers in sqrt
    if offsets_st is None:
        V_st = Rt - Rs  # s -> t
    else:
        V_st = Rt - Rs + offsets_st  # s -> t
    D_st = torch.sqrt(torch.sum(V_st ** 2, dim=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_clamped(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def get_angle(R_ac, R_ab):
    """Calculate angles between atoms c -> a <- b.

    Arguments
    ---------
        R_ac: Tensor, shape = (N, 3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=-1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab, dim=-1).norm(dim=-1)  # shape = (N,)
    y = y.clamp(min=1e-9)  # Avoid NaN gradient for y = (0,0,0)

    angle = torch.atan2(y, x)
    return angle


def vector_rejection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N, 3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def get_projected_angle(R_ab, P_n, eps=1e-4):
    """
    Project the vector R_ab onto a plane with normal vector P_n,
    then calculate the angle w.r.t. the (x [cross] P_n),
    or (y [cross] P_n) if the former would be ill-defined/numerically unstable.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.
        eps: float
            Norm of projection below which to use the y-axis instead of x.

    Returns
    -------
        angle_ab: Tensor, shape = (N)
            Angle on plane w.r.t. x- or y-axis.
    """
    R_ab_proj = torch.cross(R_ab, P_n, dim=-1)

    # Obtain axis defining the angle=0
    x = P_n.new_tensor([[1, 0, 0]]).expand_as(P_n)
    zero_angle = torch.cross(x, P_n, dim=-1)

    use_y = torch.norm(zero_angle, dim=-1) < eps
    P_n_y = P_n[use_y]
    y = P_n_y.new_tensor([[0, 1, 0]]).expand_as(P_n_y)
    y_cross = torch.cross(y, P_n_y, dim=-1)
    zero_angle[use_y] = y_cross

    angle = get_angle(zero_angle, R_ab_proj)

    # Flip sign of angle if necessary to obtain clock-wise angles
    cross = torch.cross(zero_angle, R_ab_proj, dim=-1)
    flip_sign = torch.sum(cross * P_n, dim=-1) < 0
    angle[flip_sign] = -angle[flip_sign]

    return angle


def mask_neighbors(neighbors, edge_mask):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
    return neighbors


def get_neighbor_order(num_atoms, index, atom_distance):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    """
    device = index.device

    # Get sorted index and inverse sorting
    # Necessary for index_sort_map
    index_sorted, index_order = torch.sort(index)
    index_order_inverse = torch.argsort(index_order)

    # Get number of neighbors
    ones = index_sorted.new_ones(1).expand_as(index_sorted)
    num_neighbors = segment_coo(ones, index_sorted, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index_sorted * max_num_neighbors
            + torch.arange(len(index_sorted), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Offset index_sort so that it indexes into index_sorted
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # Create indices specifying the order in index_sort
    order_peratom = torch.arange(max_num_neighbors, device=device)[
                    None, :
                    ].expand_as(mask_finite)
    order_peratom = torch.masked_select(order_peratom, mask_finite)

    # Re-index to obtain order value of each neighbor in index_sorted
    order = torch.zeros(len(index), device=device, dtype=torch.long)
    order[index_sort] = order_peratom

    return order[index_order_inverse]


def get_inner_idx(idx, dim_size):
    """
    Assign an inner index to each element (neighbor) with the same index.
    For example, with idx=[0 0 0 1 1 1 1 2 2] this returns [0 1 2 0 1 2 3 0 1].
    These indices allow reshape neighbor indices into a dense matrix.
    idx has to be sorted for this to work.
    """
    ones = idx.new_ones(1).expand_as(idx)
    num_neighbors = segment_coo(ones, idx, dim_size=dim_size)
    inner_idx = ragged_range(num_neighbors)
    return inner_idx


def get_edge_id(edge_idx, cell_offsets, num_atoms):
    cell_basis = cell_offsets.max() - cell_offsets.min() + 1
    cell_id = (
        (
                cell_offsets
                * cell_offsets.new_tensor([[1, cell_basis, cell_basis ** 2]])
        )
        .sum(-1)
        .long()
    )
    edge_id = edge_idx[0] + edge_idx[1] * num_atoms + cell_id * num_atoms ** 2
    return edge_id


def get_pbc_distances_oc(
        pos,
        edge_index,
        cell,
        cell_offsets,
        neighbors,
        return_offsets=False,
        return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[
        distances != 0
        ]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out
