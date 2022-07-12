import os
from typing import Sequence, Any
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from data.utils import get_mnist_dataloader


class Siren(nn.Module):
    pass

@jax.jit
def train_step(state, image):
    represent = None
    return state, represent

def train(num_iters, image, state, rngs, writer):
    for i in range(1, num_iters+1):
        state, represent = train_step(state, image)


def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rngs = random.PRNGKey(seed=123)
    dummy_x = jnp.ones((1, 32, 32, 1))

    state = siren.init(rngs, dummy_x)
    tx = optax.adam(learning_rate=1e-4)
    
    train_state = TrainState.create(apply_fn=model.apply,
                                    params=state['params'],
                                    tx=tx)

    train_loader, _ = get_mnist_dataloader(batch_size=8)
    loader = iter(train_loader)
    image, _ = next(loader)

    save_dir = 'exp_results/NIR/siren_mnist/version_0'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    
    
    train(5, train_loader, test_loader, train_state, rngs, writer)
    

if __name__ == '__main__':
    main()