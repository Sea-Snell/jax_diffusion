from __future__ import annotations
from collections import namedtuple
from functools import partial
from typing import List, Optional
from micro_config import ConfigScript, MetaConfig, ConfigScriptModel
from dataclasses import dataclass
import mnist
import haiku as hk
import jax
import jax.numpy as jnp
import pickle as pkl
import optax
from src import MLP, MNISTData, mnist_per_number_evaluator

@dataclass
class RNGSeed(ConfigScript):
    value: int

    def unroll(self, metaconfig: MetaConfig):
        return jax.random.PRNGKey(self.value)
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)

@dataclass
class RNGSplit(ConfigScript):
    seed: ConfigScript
    n_splits: int

    def unroll(self, metaconfig: MetaConfig):
        rng = self.seed.unroll(metaconfig)
        if self.n_splits == 0:
            return rng
        for _ in range(self.n_splits):
            rng, new_rng = jax.random.split(rng)
        return new_rng

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
class MLPConfig(ConfigScript):
    shapes: List[int]
    dropout: float
    rng: ConfigScript
    checkpoint_path: Optional[str]=None

    def unroll(self, metaconfig: MetaConfig):
        model = hk.multi_transform_with_state(partial(MLP.multi_transform_f, self.shapes[1:], self.dropout))
        rng = self.rng.unroll(metaconfig)
        if self.checkpoint_path is None:
            params, state = model.init(rng, jnp.zeros((1, self.shapes[0],)), is_training=True)
        else:
            if metaconfig.verbose:
                print('loading model state from: %s' % metaconfig.convert_path(self.checkpoint_path))
            with open(metaconfig.convert_path(self.checkpoint_path), 'rb') as f:
                params, state = pkl.load(f)
            if metaconfig.verbose:
                print('loaded.')
        return model.apply, params, state

@dataclass
class MNISTPerNumberEvaluatorConfig(ConfigScript):
    rng: ConfigScript

    def unroll(self, metaconfig: MetaConfig):
        rng = self.rng.unroll(metaconfig)
        init, apply, can_jit = mnist_per_number_evaluator()
        eval_state = init(rng)
        return apply, eval_state, can_jit

@dataclass
class AdamWConfig(ConfigScript):
    lr: float
    weight_decay: float
    grad_accum_steps: int=1
    model: Optional[ConfigScript]=None
    state_path: Optional[str]=None

    def __post_init__(self):
        assert (self.model is not None) or (self.state_path is not None)

    def unroll(self, metaconfig: MetaConfig):
        optimizer = optax.adamw(self.lr, weight_decay=self.weight_decay)
        if self.grad_accum_steps > 1:
            optimizer = optax.MultiSteps(optimizer, 
                                         self.grad_accum_steps, 
                                         use_grad_mean=True)
        if self.state_path is not None:
            if metaconfig.verbose:
                print('loading optimizer state from: %s' % metaconfig.convert_path(self.state_path))
            with open(metaconfig.convert_path(self.state_path), 'rb') as f:
                optim_state = pkl.load(f)
                if isinstance(optim_state, optax.MultiStepsState) and self.grad_accum_steps == 1:
                    optim_state = optim_state.inner_opt_state
                elif (not isinstance(optim_state, optax.MultiStepsState)) and self.grad_accum_steps > 1:
                    _, params, _ = self.model.unroll(metaconfig)
                    new_optim_state = optimizer.init(params)
                    optim_state = new_optim_state._replace(inner_opt_state=optim_state)
            if metaconfig.verbose:
                print('loaded.')
        elif self.model is not None:
            _, params, _ = self.model.unroll(metaconfig)
            optim_state = optimizer.init(params)
        else:
            raise NotImplementedError
        return optimizer.update, optim_state

