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
    latents: int

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


        mean_x = nn.Conv(features=self.latents, 
                        kernel_size=(4,4),
                        strides=(1,1),
                        padding='VALID',
                        use_bias=False,
                        name='conv_mean')(x)
        logvar_x = nn.Conv(features=self.latents, 
                        kernel_size=(4,4),
                        strides=(1,1),
                        padding='VALID',
                        use_bias=False,
                        name='conv_logvar')(x)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=64*2, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x = nn.ConvTranspose(features=64*2, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        
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
    latents: int

    def setup(self):
        self.encoder = Encoder(self.training, latents=self.latents)
        self.decoder = Decoder(self.training)

    def generate(self, z):
        return jnp.tanh(self.decoder(z))

    def reparametrize(self, rng, mean, logvar):
        std = jnp.exp(0.5*logvar)
        noise = random.normal(rng, logvar.shape)
        return mean + noise * std

    def __call__(self, x, rng):
        mean, logvar = self.encoder(x)
        z = self.reparametrize(rng, mean, logvar)
        x = self.decoder(z)
        return jnp.tanh(x), mean, logvar


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def bce_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def train_step(state, x, z_rng):
    def loss_fn(params, variables):
        (recon, mean, logvar), variables = state.apply_fn({'params':params, 'batch_stats': variables['batch_stats']},
                                                            x, z_rng,
                                                            mutable=['batch_stats'])
        
        bce_loss = bce_with_logits(recon, x).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss, (variables, bce_loss, kld_loss, recon)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (variables, bce_loss, kld_loss, recon)), grads = grad_fn(state.params, {'batch_stats':state.batch_stats})
    state = state.apply_gradients(grads=grads, batch_stats=variables["batch_stats"])
    return state, bce_loss, kld_loss, recon

def train(num_epochs, train_loader, state, rng, writer):
    for epoch in range(1, num_epochs+1):
        for i, (x, _) in enumerate(train_loader):
            rng, z_rng = random.split(rng)

            x = jnp.expand_dims(x, axis=-1)
            x = x * 2. - 1.
            state, bce_loss, kld_loss, recon = train_step(state, x, z_rng)

            if (i+1)%500 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} |\
                     BCE Loss {bce_loss:.3f} | KLD Loss {kld_loss:.3f}')

                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss/BCE', to_np(bce_loss), num_steps)
                writer.add_scalar('Loss/KLD', to_np(kld_loss), num_steps)
                writer.add_images('Recon', (to_np(recon)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('Input', (to_np(x)+1.)/2., num_steps, dataformats='NHWC')

def main():
    rng = random.PRNGKey(seed=123)
    autoencoder = Autoencoder(training=True, latents=128)
    dummy_x = jnp.ones((1,32,32,1), dtype=jnp.float32)
    state = autoencoder.init(rng, dummy_x, rng)

    tx = optax.adam(learning_rate=1e-4)
    train_state = TrainStateBN.create(apply_fn=autoencoder.apply,
                                    params=state["params"],
                                    tx=tx,
                                    batch_stats=state["batch_stats"])

    train_loader, test_loader = get_mnist_dataloader(batch_size=8)
    
    save_dir = 'exp_results/MNIST/vae/'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    rng, step_rng = random.split(rng)
    train(5, train_loader, train_state, rng, writer)
    

if __name__ == '__main__':
    main()