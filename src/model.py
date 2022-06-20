from collections import namedtuple
from typing import List
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from torch.utils.data import Dataset
from logs import LogTuple

class DiffusionModel(hk.Module):
    def __init__(self, beta: List[float]):
        super().__init__(name='DiffusionModel')
        self.T = len(beta)
        self.beta = jnp.array(beta)
        self.alpha = 1 - self.beta
        self.alpha_hat = jnp.cumprod(self.alpha)
        

