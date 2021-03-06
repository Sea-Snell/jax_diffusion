from __future__ import annotations
from functools import partial
from typing import List
from micro_config import ConfigScript, MetaConfig
from haiku_configs import ConfigScriptModel, ConfigScriptOptim, ModelConfigReturn
from dataclasses import dataclass
import mnist
import haiku as hk
import jax.numpy as jnp
import optax
from src import MLP, MNISTCNN, MNISTData
import os

project_root = os.path.dirname(__file__)

@dataclass
class MNISTDataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        if self.split == 'train':
            imgs = mnist.train_images()
            labels = mnist.train_labels()
        elif self.split == 'test':
            imgs = mnist.test_images()
            labels = mnist.test_labels()
        else:
            raise NotImplementedError
        return MNISTData(imgs=imgs, labels=labels)

@dataclass
class MLPConfig(ConfigScriptModel):
    shapes: List[int]
    dropout: float

    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = hk.multi_transform_with_state(partial(MLP.multi_transform_f, self.shapes[1:], self.dropout))
        return ModelConfigReturn(model, (jnp.zeros((1, self.shapes[0],)),), {'train': True})

@dataclass
class MNISTCNNConfig(ConfigScriptModel):
    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = hk.multi_transform_with_state(MNISTCNN.multi_transform_f)
        return ModelConfigReturn(model, (jnp.zeros((1, 28*28,)),), {'train': True})

@dataclass
class AdamWConfig(ConfigScriptOptim):
    lr: float
    weight_decay: float

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        return optax.adamw(self.lr, weight_decay=self.weight_decay)
