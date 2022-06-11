# type: ignore

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
from data.utils import get_denoise_mnist_dataloader as get_mnist_dataloader

def to_np(a):
    return np.asarray(a)

class TrainStateBN(TrainState):
    batch_stats: Any

class Encoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        # first CR-P (32x32 -> 16x16)
        x1 = nn.Conv(features=64, kernel_size=(3,3), 
                    strides=(1,1), padding='SAME', use_bias=False)(x)
        x1 = nn.leaky_relu(x1, negative_slope=0.2)

        x1_pool = nn.pooling.max_pool(x1, window_shape=(2,2), strides=(2,2))

        # second CBR-P (16x16 -> 8x8)
        x2 = nn.Conv(features=64*2, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x1_pool)
        x2 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x2)
        x2 = nn.leaky_relu(x2, negative_slope=0.2)

        x2_pool = nn.pooling.max_pool(x2, window_shape=(2,2), strides=(2,2))

        # third CBR-P (8x8 -> 4x4)
        x3 = nn.Conv(features=64*4, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x2_pool)
        x3 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x3)
        x3 = nn.leaky_relu(x3, negative_slope=0.2)

        x3_pool = nn.pooling.max_pool(x3, window_shape=(2,2), strides=(2,2))
        
        # last CBR ( 4x4 -> 4x4 ) 
        x4 = nn.Conv(features=64*8, kernel_size=(3,3),
                     strides=(1,1), padding='SAME', use_bias=False)(x3_pool)
        x4 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x4)
        x_out = nn.leaky_relu(x4, negative_slope=0.2)

        return x_out, [x2_pool, x1_pool]

class Decoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x, skip_feats):
        # first CBR (4x4 -> 8x8)
        x3 = nn.ConvTranspose(features=64*2, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x3 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x3)
        x3 = nn.leaky_relu(x3, negative_slope=0.2)
        x3 += skip_feats[0]

        # second CBR (8x8->16x16)
        x2 = nn.ConvTranspose(features=64, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x3)
        x2 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x2)
        x2 = nn.leaky_relu(x2, negative_slope=0.2)
        x2 += skip_feats[1]

        # last CR (16x16->32x32)
        x_out = nn.ConvTranspose(features=1, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x2)
        # x2 = nn.leaky_relu(x2, negative_slope=0.2)
        return x_out
    
class Unet(nn.Module):
    training: bool

    def setup(self):
        self.encoder = Encoder(self.training)
        self.decoder = Decoder(self.training)

    def __call__(self, x):
        x_feats, skip_feats = self.encoder(x)
        x = self.decoder(x_feats, skip_feats)
        return jnp.tanh(x)


@jax.vmap
def MSE_loss(y_pred, y_true):
    return ((y_pred-y_true)**2).sum()**0.5

@jax.jit
def train_step(state, x, y):
    def loss_fn(params, variables):
        recon, variables = state.apply_fn({'params':params, 'batch_stats': variables['batch_stats']},
                                x,
                                mutable=['batch_stats'])
        loss = MSE_loss(recon, y).mean()
        return loss, (variables, recon)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (variables, recon)), grads = grad_fn(state.params, {'batch_stats':state.batch_stats})
    state = state.apply_gradients(grads=grads, batch_stats=variables["batch_stats"])
    return state, loss, recon

def train(num_epochs, train_loader, state, writer):
    for epoch in range(1, num_epochs+1):
        for i, (x, y) in enumerate(train_loader):
            
            process = lambda in_: jnp.expand_dims(in_, axis=-1) * 2. - 1.
            x,y = list(map(process, [x,y]))
            
            state, loss, recon = train_step(state, x, y)

            if (i+1)%500 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} | Loss {loss:.3f}')

                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss', to_np(loss), num_steps)
                writer.add_images('Denoised', (to_np(recon)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('Label', (to_np(y)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('Input', (to_np(x)+1.)/2., num_steps, dataformats='NHWC')

def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rng = random.PRNGKey(seed=123)
    unet = Unet(training=True)
    dummy_x = jnp.ones((1,32,32,1), dtype=jnp.float32)
    state = unet.init(rng, dummy_x)

    tx = optax.adam(learning_rate=1e-4)
    train_state = TrainStateBN.create(apply_fn=unet.apply,
                                    params=state["params"],
                                    tx=tx,
                                    batch_stats=state["batch_stats"])

    train_loader, test_loader = get_mnist_dataloader(batch_size=32)
    
    save_dir = 'exp_results/MNIST/denoise/unet'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    train(5, train_loader, train_state, writer)
    
if __name__ == '__main__':
    main()