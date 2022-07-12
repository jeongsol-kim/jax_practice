#type: ignore

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

def to_np(a):
    return np.asarray(a)

class TrainStateBN(TrainState):
    batch_stats: Any

class Encoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        # first CR-P (32x32 -> 16x16)
        x = nn.Conv(features=64, kernel_size=(3,3), 
                    strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.pooling.max_pool(x, window_shape=(2,2), strides=(2,2))

        # second CBR-P (16x16 -> 8x8)
        x = nn.Conv(features=64*2, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.pooling.max_pool(x, window_shape=(2,2), strides=(2,2))

        # third CBR-P (8x8 -> 4x4)
        x = nn.Conv(features=64*4, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.pooling.max_pool(x, window_shape=(2,2), strides=(2,2))

        return x

class Decoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=64*2, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.ConvTranspose(features=64, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.ConvTranspose(features=1, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        return x

class Autoencoder(nn.Module):
    training: bool

    def setup(self):
        self.encoder = Encoder(self.training)
        self.decoder = Decoder(self.training)

    def __call__(self, x):
        x = self.decoder(self.encoder(x))
        return jnp.tanh(x)


@jax.vmap
def MSE_loss(y_pred, y_true):
    return ((y_pred-y_true)**2).sum()**0.5

@jax.jit
def train_step(state, x):
    def loss_fn(params, variables):
        recon, variables = state.apply_fn({'params':params, 'batch_stats': variables['batch_stats']},
                                x,
                                mutable=['batch_stats'])
        
        loss = MSE_loss(recon, x).mean()
        return loss, (variables, recon)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (variables, recon)), grads = grad_fn(state.params, {'batch_stats':state.batch_stats})
    state = state.apply_gradients(grads=grads, batch_stats=variables["batch_stats"])
    return state, loss, recon

def train(num_epochs, train_loader, state, writer):
    for epoch in range(1, num_epochs+1):
        for i, (x, _) in enumerate(train_loader):
            x = jnp.expand_dims(x, axis=-1)
            x = x * 2. - 1.
            state, loss, recon = train_step(state, x)

            if (i+1)%500 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} | Loss {loss:.3f}')

                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss', to_np(loss), num_steps)
                writer.add_images('Recon', (to_np(recon)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('Input', (to_np(x)+1.)/2., num_steps, dataformats='NHWC')

def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rng = random.PRNGKey(seed=123)
    autoencoder = Autoencoder(training=True)
    dummy_x = jnp.ones((1,32,32,1), dtype=jnp.float32)
    state = autoencoder.init(rng, dummy_x)

    tx = optax.adam(learning_rate=1e-4)
    train_state = TrainStateBN.create(apply_fn=autoencoder.apply,
                                    params=state["params"],
                                    tx=tx,
                                    batch_stats=state["batch_stats"])

    train_loader, test_loader = get_mnist_dataloader(batch_size=8)
    
    save_dir = 'exp_results/MNIST/autoencoder/'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    train(5, train_loader, train_state, writer)
    
if __name__ == '__main__':
    main()