"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import bisect
import os
import warnings

import ase.io.trajectory
import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Subset, ConcatDataset, Dataset

from grannfield.utils.utils import read_deprecated_database

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

from tqdm import tqdm


class MaterialsDataError(Exception):
    pass


class MaterialsData(Dataset):
    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        available_properties=None,
        load_only=None,
        units=None,
    ):

        # checks
        if not dbpath.endswith('.db'):
            raise MaterialsDataError(
                'Invalid dbpath! Please make sure to add the file extension .db to '
                'your dbpath.'
            )

        # database
        self.dbpath = dbpath

        # check if database is deprecated:
        if self._is_deprecated():
            self._deprecation_update()

        self._load_only = load_only
        self._available_properties = self._get_available_properties(
            available_properties
        )

        if units is None:
            units = [1.0] * len(self.available_properties)
        self.units = dict(zip(self.available_properties, units))

        if len(units) != len(self.available_properties):
            raise MaterialsDataError(
                'The length of available properties and units does not match!'
            )

        self.additional_properties = {}

    @property
    def available_properties(self):
        return self._available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.available_properties
        return self._load_only

    def get_key(self, idx, key):
        with connect(self.dbpath) as conn:
            row = conn.get(idx+1)
            id = row[key]
        return id

    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        with connect(self.dbpath) as conn:
            if key is None:
                return conn.metadata
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    def set_metadata(self, metadata=None, **kwargs):
        """
        Sets the metadata dictionary of the ASE db.

        Args:
            metadata (dict): dictionary of metadata for the ASE db
            kwargs: further key-value pairs for convenience
        """

        # merge all metadata
        if metadata is not None:
            kwargs.update(metadata)

        with connect(self.dbpath) as conn:
            conn.metadata = kwargs

    def update_metadata(self, data):
        with connect(self.dbpath) as conn:
            metadata = conn.metadata
        metadata.update(data)
        self.set_metadata(metadata)

    # get atoms and properties
    def get_properties(self, idx, load_only=None):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index
            load_only (sequence or None): subset of available properties to load

        Returns:
            at (ase.Atoms): atoms object
            properties (dict): dictionary with molecular properties

        """
        # use all available properties if nothing is specified
        if load_only is None:
            load_only = self.available_properties

        # read from ase-database
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in load_only:
            properties[pname] = row.data[pname]

        for kname in row.key_value_pairs.keys():
            properties[kname] = row.key_value_pairs[kname]

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            idx=idx,
            additional_properties = self.additional_properties,
            output=properties,
        )

        return at, properties

    def add_property(self, property_name, values):
        assert len(values) == len(self)
        self.additional_properties[property_name] = values

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    # add systems
    def add_system(self, atoms, properties=dict(), key_value_pairs=dict()):
        """
        Add atoms data to the dataset.

        Args:
            atoms (ase.Atoms): system composition and geometry
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, properties, key_value_pairs)

    def add_systems(self, atoms_list, property_list=None, key_value_pairs_list=None):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list (list of ase.Atoms): system composition and geometry
            property_list (list): Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset.

        """
        # build empty dicts if property/kv_pairs list is None
        if property_list is None:
            property_list = [dict() for _ in range(len(atoms_list))]

        if key_value_pairs_list is None:
            key_value_pairs_list = [dict() for _ in range(len(atoms_list))]

        # write systems to database
        with connect(self.dbpath) as conn:
            for at, prop, kv_pair in zip(
                atoms_list, property_list, key_value_pairs_list
            ):
                self._add_system(conn, at, prop, kv_pair)

    # __functions__
    def __len__(self):
        with connect(self.dbpath) as conn:
            return conn.count()

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties['idx'] = np.array([idx], dtype=int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatMaterialsData([self, other])

    # private methods
    def _add_system(self, conn, atoms, properties=dict(), key_value_pairs=dict()):
        """
        Write systems to the database. Floats, ints and np.ndarrays without dimension are transformed to np.ndarrays with dimension 1.

        """
        data = {}
        # add available properties to database
        for pname in self.available_properties:
            try:
                data[pname] = properties[pname]
            except:
                raise MaterialsDataError('Required property missing:' + pname)

        # transform to np.ndarray
        data = numpyfy_dict(data)

        conn.write(atoms, data=data, key_value_pairs=key_value_pairs)

    def _get_available_properties(self, properties):
        """
        Get available properties from argument or database.

        Returns:
            (list): all properties of the dataset
        """
        # read database properties
        if os.path.exists(self.dbpath) and len(self) != 0:
            with connect(self.dbpath) as conn:
                atmsrw = conn.get(1)
                db_properties = list(atmsrw.data.keys())
        else:
            db_properties = None

        # use the provided list
        if properties is not None:
            if db_properties is None or set(db_properties) == set(properties):
                return properties

            # raise error if available properties do not match database
            raise MaterialsDataError(
                'The available_properties {} do not match the '
                'properties in the database {}!'.format(properties, db_properties)
            )

        # return database properties
        if db_properties is not None:
            return db_properties

        raise MaterialsDataError(
            'Please define available_properties or set db_path to an existing database!'
        )

    def _is_deprecated(self):
        """
        Check if database is deprecated.

        Returns:
            (bool): True if ase db is deprecated.
        """
        # check if db exists
        if not os.path.exists(self.dbpath):
            return False

        # get properties of first atom
        with connect(self.dbpath) as conn:
            data = conn.get(1).data

        # check byte style deprecation
        if True in [pname.startswith('_dtype_') for pname in data.keys()]:
            return True
        # fallback for properties stored directly in the row
        if True in [type(val) != np.ndarray for val in data.values()]:
            return True

        return False

    def _deprecation_update(self):
        """
        Update deprecated database to a valid ase database.
        """
        warnings.warn(
            'The database is deprecated and will be updated automatically. '
            'The old database is moved to {}.deprecated!'.format(self.dbpath)
        )

        # read old database
        (
            atoms_list,
            properties_list,
            key_value_pairs_list,
        ) = read_deprecated_database(self.dbpath)
        metadata = self.get_metadata()

        # move old database
        os.rename(self.dbpath, self.dbpath + '.deprecated')

        # write updated database
        self.set_metadata(metadata=metadata)
        with connect(self.dbpath) as conn:
            for atoms, properties, key_value_pairs in tqdm(
                zip(atoms_list, properties_list, key_value_pairs_list),
                'Updating new database',
                total=len(atoms_list),
            ):
                conn.write(
                    atoms,
                    data=numpyfy_dict(properties),
                    key_value_pairs=key_value_pairs,
                )


class ConcatMaterialsData(ConcatDataset):
    r"""
    Dataset as a concatenation of multiple atomistic datasets.
    Args:
        datasets (sequence): list of datasets to be concatenated
    """

    def __init__(self, datasets):
        # checks
        for dataset in datasets:
            if not any(
                [
                    isinstance(dataset, dataset_class)
                    for dataset_class in [MaterialsData, MaterialsDataSubset, ConcatDataset]
                ]
            ):
                raise MaterialsDataError(
                    '{} is not an instance of MaterialsData, MaterialsDataSubset or '
                    'ConcatMaterialsData!'.format(dataset)
                )
        super(ConcatMaterialsData, self).__init__(datasets)
        self._load_only = None

    @property
    def load_only(self):
        if self._load_only:
            return self._load_only
        load_onlys = [set(dataset.load_only) for dataset in self.datasets]
        return list(load_onlys[0].intersection(*load_onlys[1:]))

    @property
    def available_properties(self):
        all_available_properties = [
            set(dataset.available_properties) for dataset in self.datasets
        ]
        return list(
            all_available_properties[0].intersection(*all_available_properties[1:])
        )

    def get_properties(self, idx, load_only=None):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length'
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_properties(sample_idx, load_only)

    def get_atoms(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length'
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_atoms(sample_idx)

    def set_load_only(self, load_only):
        # check if properties are available
        for pname in load_only:
            if pname not in self.available_properties:
                raise MaterialsDataError(
                    'The property {} is not an available property and can therefore '
                    'not be loaded!'.format(pname)
                )

        # update load_only parameter
        self._load_only = list(load_only)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties['idx'] = np.array([idx], dtype=int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatMaterialsData([self, other])


class MaterialsDataSubset(Subset):
    r"""
    Subset of an atomistic dataset at specified indices.
    Arguments:
        dataset (torch.utils.data.Dataset): atomistic dataset
        indices (sequence): subset indices
    """

    def __init__(self, dataset, indices):
        super(MaterialsDataSubset, self).__init__(dataset, indices)
        self._load_only = None

    @property
    def available_properties(self):
        return self.dataset.available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.dataset.load_only
        return self._load_only

    def get_atomref(self, properties):
        return self.dataset.get_atomref(properties)

    def get_properties(self, idx, load_only=None):
        return self.dataset.get_properties(self.indices[idx], load_only)

    def get_atoms(self, idx):
        return self.dataset.get_atoms(self.indices[idx])

    def set_load_only(self, load_only):
        # check if properties are available
        for pname in load_only:
            if pname not in self.available_properties:
                raise MaterialsDataError(
                    'The property {} is not an available property and can therefore '
                    'not be loaded!'.format(pname)
                )

        # update load_only parameter
        self._load_only = list(load_only)

    # deprecated
    def create_subset(self, subset):
        from grannfield.data.data import create_subset
        return create_subset(self, subset)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["idx"] = np.array([idx], dtype=int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatMaterialsData([self, other])


def _convert_atoms(
        atoms,
        idx=None,
        additional_properties=None,
        output=None,
):
    """
    Helper function to convert ASE atoms object to SchNetPack input format.

    Args:
        atoms (ase.Atoms): Atoms object of molecule
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
        output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.

    """
    if additional_properties is None:
        additional_properties = {}
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    inputs['atomic_numbers'] = np.array(atoms.numbers, dtype=int)
    positions = atoms.positions.astype(np.float32)
    inputs['n_atoms'] = np.array([positions.shape[0]], dtype=int)
    inputs['positions'] = positions
    inputs['frac_positions'] = atoms.get_scaled_positions().astype(np.float32)
    inputs['volume'] = np.array([atoms.cell.volume], dtype=np.float32)

    fixed_idx = np.zeros(positions.shape[0], dtype=int)
    if hasattr(atoms, 'constraints'):
        from ase.constraints import FixAtoms

        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_idx[constraint.index] = 1

    inputs['fixed'] = fixed_idx
    inputs['tags'] = atoms.get_tags()

    if additional_properties is not {}:
        for key in additional_properties.keys():
            inputs[key] = np.array([additional_properties[key][idx]], dtype=np.float32)

    # Get cells
    inputs['cell'] = np.array([atoms.cell.array], dtype=np.float32)

    return inputs


def torchify_dict(data):
    """
    Transform np.ndarrays to torch.tensors.

    """
    torch_properties = {}
    for pname, prop in data.items():

        try:
            if prop.dtype in [int, np.int32, np.int64]:
                torch_properties[pname] = torch.LongTensor(prop)
            elif prop.dtype in [float, np.float32, np.float64]:
                torch_properties[pname] = torch.FloatTensor(prop.copy())
            else:
                raise MaterialsDataError(
                    'Invalid datatype {} for property {}!'.format(type(prop), pname)
                )
        except AttributeError:
            if type(prop) == str:
                pass
            elif type(prop) == int:
                torch_properties[pname] = torch.LongTensor([prop])
            else:
                torch_properties[pname] = torch.FloatTensor([prop])

    return torch_properties


def numpyfy_dict(data):
    """
    Transform floats, ints and dimensionless numpy in a dict to arrays to numpy arrays with dimenison.

    """
    for k, v in data.items():
        if type(v) in [int, float]:
            v = np.array([v])
        if v.shape == ():
            v = v[np.newaxis]
        data[k] = v
    return data


class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the GranField calculator
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
    """

    def __init__(
        self,
        device=torch.device('cpu'),
    ):
        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms)
        inputs = torchify_dict(inputs)
        n_i = inputs['n_atoms'].item()
        inputs['batch'] = torch.LongTensor([0]).expand(n_i)
        fixed_idx = torch.zeros(n_i)
        if hasattr(atoms, "constraints"):
            from ase.constraints import FixAtoms

            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    fixed_idx[constraint.index] = 1
        inputs['fixed'] = fixed_idx

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)

        return inputs
