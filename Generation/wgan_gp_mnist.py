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


class Generator(nn.Module):
    """
    Taken from https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb
    """
    training: bool

    @nn.compact
    def __call__(self, z):
        x = nn.ConvTranspose(features=64*8, kernel_size=(4, 4),
                             strides=(1, 1), padding='VALID', use_bias=False)(z)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=64*4, kernel_size=(4, 4),
                             strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=64*2, kernel_size=(4, 4),
                             strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=64, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=1, kernel_size=(
            4, 4), strides=(1, 1), padding='SAME', use_bias=False)(x)
        return jnp.tanh(x)

class Discriminator(nn.Module):
    """
    Modified from https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb.
    BatchNorm should not be used for critics.    
    """
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*2, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*4, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*8, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=1, kernel_size=(
            1, 1), strides=(4, 4), padding='VALID', use_bias=False)(x)
        x = jnp.reshape(x, [x.shape[0], -1])

        return x

@jax.jit
def train_step(state_g, state_d, noise, real, gp_rng):
    # we assume that n_critics=1

    def gradient_panelty(interpolate, params_d):
        inter_logits = Discriminator(training=True).apply(
                        {'params':params_d},
                        interpolate)

        return inter_logits.mean()

    def loss_g_fn(params_g, params_d, variable_g):
        fake, variable_g = Generator(training=True).apply(
            {'params':params_g, 'batch_stats':variable_g["batch_stats"]},
            noise,
            mutable=['batch_stats']
        )

        fake_logits = Discriminator(training=True).apply(
            {'params':params_d},
            fake)

        loss_g = -fake_logits.mean()
        return loss_g, (variable_g, fake)

    def loss_d_fn(params_g, params_d, variable_g):
        fake, variable_g = Generator(training=True).apply(
            {'params':params_g, 'batch_stats':variable_g["batch_stats"]},
            noise,
            mutable=['batch_stats'])

        real_logits= Discriminator(training=True).apply(
            {'params':params_d},
            real)

        fake_logits= Discriminator(training=True).apply(
            {'params':params_d},
            fake)

        real_loss = real_logits.mean()
        fake_loss = fake_logits.mean()

        # gradient panelty
        alpha = random.uniform(gp_rng, shape=(fake.shape[0],1,1,1), 
                                    dtype=jnp.float32, minval=0, maxval=1)
        alpha = jnp.tile(alpha, (1, fake.shape[1], fake.shape[2], fake.shape[3]))
        interpolate = alpha * real + (1-alpha) * fake

        gp = jax.grad(gradient_panelty, argnums=0)(interpolate, params_d)
        gp = jnp.reshape(gp, (gp.shape[0], -1))
        gp_norm = jnp.sqrt((gp**2).sum(axis=1) + 1e-12)
        gp_norm = ((1-gp_norm)**2).mean()

        loss_d = fake_loss - real_loss + 10*gp_norm

        return loss_d, (variable_g, fake)

        
    grad_g_fn = jax.value_and_grad(loss_g_fn, argnums=0, has_aux=True)
    grad_d_fn = jax.value_and_grad(loss_d_fn, argnums=1, has_aux=True)

    variable_g = {'batch_stats': state_g.batch_stats}

    # update discriminator
    (loss_d, (variable_g, fake)), grads_d = grad_d_fn(state_g.params, 
                                                      state_d.params, 
                                                      variable_g)
    state_d = state_d.apply_gradients(grads=grads_d)

    # update generator
    (loss_g, (variable_g, fake)), grads_g = grad_g_fn(state_g.params, 
                                                      state_d.params, 
                                                      variable_g)

    state_g = state_g.apply_gradients(grads=grads_g, batch_stats=variable_g["batch_stats"])
    
    return state_g, state_d, loss_g, loss_d, fake

def train(num_epochs, train_loader, eval_loader, state_g, state_d, rng, writer):
    for epoch in range(1, num_epochs+1):
        for i, (x, _) in enumerate(train_loader):
            rng, step_rng = random.split(rng)
            noise = jax.random.normal(step_rng, shape=(x.shape[0], 1, 1, 100))

            x = jnp.expand_dims(x, axis=-1)
            x = x * 2. - 1.

            rng, gp_rng = random.split(rng)
            state_g, state_d, loss_g, loss_d, fake = \
                train_step(state_g, state_d, noise, x, gp_rng)

            if (i+1)%50 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} |\
                     Loss_G {loss_g:.3f} | Loss_D {loss_d:.3f}')

                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss/generator', to_np(loss_g), num_steps)
                writer.add_scalar('Loss/discriminator', to_np(loss_d), num_steps)
                writer.add_images('Fake', (to_np(fake)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('Real', (to_np(x)+1.)/2., num_steps, dataformats='NHWC')
        
def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rngs = random.PRNGKey(seed=123)
    generator = Generator(training=True)
    discriminator = Discriminator(training=True)
    dummy_z = jnp.ones((1, 1, 1, 100)) # NCHW
    dummy_x = jnp.ones((1, 32, 32, 1))
    g_state = generator.init(rngs, dummy_z)
    d_state = discriminator.init(rngs, dummy_x)

    g_tx = optax.adam(learning_rate=1e-4)
    d_tx = optax.adam(learning_rate=1e-4)

    g_train_state = TrainStateBN.create(apply_fn=generator.apply,
                                        params=g_state["params"], 
                                        tx=g_tx, 
                                        batch_stats=g_state["batch_stats"])
    
    d_train_state = TrainState.create(apply_fn=discriminator.apply, 
                                      params=d_state["params"], 
                                      tx=d_tx)

    train_loader, test_loader = get_mnist_dataloader(batch_size=8)

    save_dir = 'exp_results/MNIST/generation/wgan_gp/version_1/'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    train(5, train_loader, test_loader, g_train_state, d_train_state, rngs, writer)
    

if __name__ == '__main__':
    main()