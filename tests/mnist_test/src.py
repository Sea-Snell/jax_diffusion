from collections import namedtuple
from typing import List
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from torch.utils.data import Dataset
from logs import LogTuple

class MNISTData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    @staticmethod
    def collate(items):
        imgs, labels = zip(*items)
        imgs, labels = jnp.array(np.stack(imgs, axis=0)), jnp.array(np.stack(labels, axis=0))
        imgs, labels = jnp.reshape(imgs, (-1, 28*28)), jax.nn.one_hot(labels, 10)
        return imgs, labels

MLP_transformed = namedtuple('MLP_transformed', ['forward', 'loss'])

class MLP(hk.Module):
    def __init__(self, output_shapes: List[int], dropout_rate: float, name='mlp'):
        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.sequence = [
            hk.Linear(output_size=output_shapes[i], name='linear_%d' % (i)) for i in range(len(output_shapes))
        ]
    
    @classmethod
    def multi_transform_f(cls, *args, **kwargs):
        model = cls(*args, **kwargs)
        return model.__call__, MLP_transformed(model.__call__, model.loss)
    
    def __call__(self, x, is_training):
        dropout_rate = self.dropout_rate if is_training else 0.0
        for layer in self.sequence[:-1]:
            x = layer(x)
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            x = jax.nn.relu(x)
        x = self.sequence[-1](x)
        return x
    
    def loss(self, x, y, is_training):
        n = y.shape[0]
        predictions = self(x, is_training)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, (logs, [],)

evaluator_transformed = namedtuple('Evaluator', ['init', 'apply', 'can_jit'])

def mnist_per_number_evaluator():
    def init(rng):
        return None

    def apply(model, params, model_state, eval_state, rng, input_items):
        x, y = input_items
        outputs, model_state = model.forward(params, model_state, rng, x, is_training=False)
        predictions = jnp.argmax(outputs, axis=1)
        y = jnp.argmax(y, axis=1)
        logs = []
        for i in range(10):
            retrived = (predictions == i).astype(jnp.float32)
            relevant = (y == i).astype(jnp.float32)
            precision = LogTuple((retrived * relevant).sum() / jnp.maximum(retrived.sum(), 1), retrived.sum())
            recall = LogTuple((retrived * relevant).sum() / jnp.maximum(relevant.sum(), 1), relevant.sum())
            f1 = (2 * precision.mean * recall.mean) / jnp.maximum(precision.mean + recall.mean, 1)
            logs.append({'precision': precision, 'recall': recall, 'f1': f1})
        return logs, model_state, eval_state

    return evaluator_transformed(init, apply, True)
