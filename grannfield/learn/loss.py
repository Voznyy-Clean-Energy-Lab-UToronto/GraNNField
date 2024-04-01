"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import torch

class LossFnError(Exception):
    pass

def build_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the modules properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            diff = batch[prop] - result[prop]
            diff = diff ** 2
            err_sq = factor * torch.mean(diff)
            loss += err_sq
        return loss

    return loss_fn


def build_rmse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the modules properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            diff = batch[prop] - result[prop]
            diff = diff ** 2
            err_sq = factor * torch.mean(diff)
            err_sq = torch.sqrt(err_sq)
            loss += err_sq
        return loss

    return loss_fn
