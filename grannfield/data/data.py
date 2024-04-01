"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

stpayu/grannfield: https://github.com/stpayu/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import os
from itertools import product
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grannfield.data.materials import MaterialsDataError, MaterialsDataSubset


def _collate(data):
    batch = {
        prop: val for prop, val in data[0].items()
    }

    for k, properties in enumerate(data):
        if k != 0:
            for prop, val in properties.items():
                batch[prop] = torch.cat(
                    (batch[prop], val), dim=0
                )

    batch_idx = []
    seq_index = 0

    for i in range(len(data)):
        n_i = data[i]['n_atoms'].item()
        batch_idx.append(torch.tensor([seq_index]).expand(n_i))
        seq_index += 1

    batch['batch'] = torch.cat(batch_idx)

    return batch


def train_test_split(
        data,
        num_train: Any = None,
        num_val: Any = None,
        num_test: Any = None,
        split_file: str = None,
        stratify_partitions: bool = False,
        num_per_partition: bool = False,
):

    if split_file is not None and os.path.exists(split_file):
        S = np.load(split_file)
        train_idx = S['train_idx'].tolist()
        val_idx = S['val_idx'].tolist()
        test_idx = S['test_idx'].tolist()
    else:
        if num_train is None or num_val is None:
            raise ValueError(
                'You have to supply either split sizes (num_train /'
                + ' num_val) or an npz file with splits.'
            )

        assert num_train + num_val + num_test <= len(
            data
        ), 'Dataset is smaller than num_train + num_val + num_test!'

        num_train = num_train if num_train > 1 else num_train * len(data)
        num_val = num_val if num_val > 1 else num_val * len(data)
        num_train = int(num_train)
        num_val = int(num_val)

        if stratify_partitions:
            partitions = data.get_metadata('partitions')
            n_partitions = len(partitions)
            if num_per_partition:
                num_train_part = num_train
                num_val_part = num_val
            else:
                num_train_part = num_train // n_partitions
                num_val_part = num_val // n_partitions

            train_idx = []
            val_idx = []
            test_idx = []
            for start, stop in partitions.values():
                idx = np.random.permutation(np.arange(start, stop))
                train_idx += idx[:num_train_part].tolist()
                val_idx += idx[num_train_part : num_train_part + num_val_part].tolist()
                test_idx += idx[num_train_part + num_val_part :].tolist()

        else:
            idx = np.random.permutation(len(data))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train : num_train + num_val].tolist()
            test_idx = idx[num_train + num_val :].tolist()

        if split_file is not None:
            np.savez(
                split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

    train = create_subset(data, train_idx)
    val = create_subset(data, val_idx)
    test = create_subset(data, test_idx)

    return train, val, test


class MaterialsLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(MaterialsLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def create_subset(dataset, indices):
    max_id = 0 if len(indices) == 0 else max(indices)
    if len(dataset) <= max_id:
        raise MaterialsDataError(
            'The subset indices do not match the total length of the dataset!'
        )
    return MaterialsDataSubset(dataset, indices)