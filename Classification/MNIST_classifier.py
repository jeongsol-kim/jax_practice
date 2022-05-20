import os
from typing import Sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

def to_np(a):
    return np.asarray(a)

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

def get_dataloader(batch_size):
    os.makedirs('data/', exist_ok=True)
    train_ds = MNIST('data/', train=True, download=True, transform=FlattenAndCast())
    test_ds = MNIST('data/', train=False, download=True, transform=FlattenAndCast())
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=numpy_collate,
                              )
    test_loader = DataLoader(test_ds, 
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=numpy_collate,
                             )

    return train_loader, test_loader

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

class LinearModel(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(x) for x in self.features]
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        x = self.layers[-1](x)
        return x

def compute_metrics(logits, labels):
    one_hot_label = jax.nn.one_hot(labels, num_classes=10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_label))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = LinearModel([128, 64, 10]).apply(
            {'params':params}, x
        )
        one_hot_label = jax.nn.one_hot(y, num_classes=10)
        loss = optax.softmax_cross_entropy(logits, one_hot_label).mean()

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, y)
    return state, metrics

@jax.jit
def eval_step(state, x, y):
    logits = LinearModel([128, 64, 10]).apply(
        {'params':state.params}, x
    )
    one_hot_label = jax.nn.one_hot(y, num_classes=10)
    metrics = compute_metrics(logits, y)
    return metrics

def train(num_epochs, train_loader, eval_loader, state, writer):
    for epoch in range(1, num_epochs+1):
        for i, (x,y) in enumerate(train_loader):
            state, metrics = train_step(state, x, y)

            if (i+1)%500 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} |\
                     Loss {metrics["loss"]:.3f} | Accuracy {metrics["accuracy"]:.1f}')
        
                test_loss, test_acc = evaluation(eval_loader, state)
                
                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss/train', to_np(metrics["loss"]), num_steps)
                writer.add_scalar('Loss/test', to_np(test_loss), num_steps)
                writer.add_scalar('Accuracy/train', to_np(metrics['accuracy']), num_steps)
                writer.add_scalar('Accuracy/test', to_np(test_acc), num_steps)


def evaluation(data_loader, state):
    loss = []
    acc = []

    for x, y in data_loader:
        metrics = eval_step(state, x, y)
        loss.append(metrics["loss"])
        acc.append(metrics["accuracy"])
    
    mean_loss = jnp.array(loss).mean()
    mean_acc = jnp.array(acc).mean()

    return mean_loss, mean_acc

def main():
    rngs = random.PRNGKey(seed=123)
    model = LinearModel(features=[128, 64, 10])
    dummy_x = jnp.ones((1, 28*28))
    params = model.init(rngs, dummy_x)['params']

    tx = optax.adam(learning_rate=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    train_loader, test_loader = get_dataloader(batch_size=8)

    save_dir = 'exp_results/MNIST/'
    writer = SummaryWriter(log_dir=save_dir)
    train(5, train_loader, test_loader, state, writer)


if __name__ == '__main__':
    main()