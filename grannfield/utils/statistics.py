"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield
"""

import torch

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']