import os
from typing import Sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from data.utils import get_flat_mnist_dataloader as get_mnist_dataloader

def to_np(a):
    return np.asarray(a)

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
        # since there is no mutable variables (eg. batch_stats),
        # apply_fn returns only the output.
        logits = state.apply_fn(
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
    logits = state.apply_fn(
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
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rngs = random.PRNGKey(seed=123)
    model = LinearModel(features=[128, 64, 10])
    dummy_x = jnp.ones((1, 28*28))
    params = model.init(rngs, dummy_x)['params']

    tx = optax.adam(learning_rate=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    train_loader, test_loader = get_mnist_dataloader(batch_size=32)

    save_dir = 'exp_results/MNIST/classification'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    train(5, train_loader, test_loader, state, writer)


if __name__ == '__main__':
    main()