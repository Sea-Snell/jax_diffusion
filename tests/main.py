from typing import List
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
import mnist
from torch.utils.data import Dataset, DataLoader
from functools import partial

class MnistData(Dataset):
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

class MLP(hk.Module):
    def __init__(self, output_shapes: List[int], name='mlp'):
        super().__init__(name=name)
        self.sequence = [
            hk.Linear(output_size=output_shapes[i], name='linear_%d' % (i)) for i in range(len(output_shapes))
        ]
    
    def __call__(self, x):
        for i, layer in enumerate(self.sequence):
            x = layer(x)
            if i != len(self.sequence) - 1:
                x = jax.nn.relu(x)
        return x

    @staticmethod
    def loss(self, params, rng, x, y):
        predictions = self(params, rng, x)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean()}
        return loss, logs

def step_fn(loss_f, optimizer, params, opt_state, rng, x, y):
    (_, logs), grads = jax.value_and_grad(loss_f, has_aux=True)(params, rng, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return logs, params, opt_state

if __name__ == "__main__":
    bsize = 32
    epochs = 10
    mlp_shape = [128, 128, 10]

    rng = jax.random.PRNGKey(42)
    # a = jnp.array([1, 2, 3])
    # print(a * a)
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_dataset = MnistData(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=0, collate_fn=MnistData.collate)

    model = hk.transform(lambda x: MLP(mlp_shape)(x))
    rng, init_rng = jax.random.split(rng)
    params = model.init(rng=init_rng, x=jnp.zeros((0, 28*28)))
    
    optimizer = optax.adamw(3e-4)
    opt_state = optimizer.init(params)

    loss_f = partial(MLP.loss, model.apply)
    step = partial(step_fn, loss_f, optimizer)
    step = jax.jit(step)

    step_count = 0
    for epoch in range(epochs):
        for x, y in train_dataloader:
            rng, step_rng = jax.random.split(rng)
            logs, params, opt_state = step(params, opt_state, step_rng, x, y)
            logs = jax.tree_map(lambda x: x.item(), logs)
            step_count += 1
            if step_count % 100 == 0:
                print(epoch, step_count, logs)
    print(epoch, step_count, logs)

        
    # print(train_images.shape, train_labels.shape)

