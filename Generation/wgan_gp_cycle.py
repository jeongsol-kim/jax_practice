#type: ignore

import os
import functools
from typing import Sequence, Any
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from data.utils import get_horse2zebra_dataloader as get_dataloader

def to_np(a):
    return np.asarray(a)

class TrainStateBN(TrainState):
    batch_stats: Any

class TrainStateCycle(TrainState):
    batch_stats_a2b: Any
    batch_stats_b2a: Any

class Encoder(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        # first CR-Pool (128x128 -> 64x64)
        x1 = nn.Conv(features=64, kernel_size=(3,3), 
                    strides=(1,1), padding='SAME', use_bias=False)(x)
        x1 = nn.leaky_relu(x1, negative_slope=0.2)
        x1_pool = nn.pooling.max_pool(x1, window_shape=(2,2), strides=(2,2))

        # second CBR-P (64x64 -> 32x32)
        x2 = nn.Conv(features=64*2, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x1_pool)
        x2 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x2)
        x2 = nn.leaky_relu(x2, negative_slope=0.2)

        x2_pool = nn.pooling.max_pool(x2, window_shape=(2,2), strides=(2,2))

        # third CBR-P (32x32 -> 16x16)
        x3 = nn.Conv(features=64*4, kernel_size=(3,3),
                    strides=(1,1), padding='SAME', use_bias=False)(x2_pool)
        x3 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x3)
        x3 = nn.leaky_relu(x3, negative_slope=0.2)

        x3_pool = nn.pooling.max_pool(x3, window_shape=(2,2), strides=(2,2))
        
        # last CBR (16x16 -> 16x16) 
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
        # first CBR (16x16 -> 32x32)
        x3 = nn.ConvTranspose(features=64*2, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x)
        x3 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x3)
        x3 = nn.leaky_relu(x3, negative_slope=0.2)
        x3 += skip_feats[0]

        # second CBR (32x32 -> 64x64)
        x2 = nn.ConvTranspose(features=64, kernel_size=(2,2),
                            strides=(2,2), padding='SAME', use_bias=False)(x3)
        x2 = nn.BatchNorm(use_running_average=not self.training,
                        momentum=0.9)(x2)
        x2 = nn.leaky_relu(x2, negative_slope=0.2)
        x2 += skip_feats[1]

        # last CR (64x64 -> 128x128)
        x_out = nn.ConvTranspose(features=3, kernel_size=(2,2),
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

class PatchGANDiscriminator(nn.Module):
    """
    Taken from https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb
    BatchNorm should not be used for critics.    
    """
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(4,4), strides=(2, 2), 
                    padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=64*2, kernel_size=(4,4),
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
    
class Generator_A2B(nn.Module):
    training:bool
    
    def setup(self):
        self.net = Unet(self.training)
        
    def __call__(self, x):
        return self.net(x)
    
class Generator_B2A(nn.Module):
    training:bool
    
    def setup(self):
        self.net = Unet(self.training)
        
    def __call__(self, x):
        return self.net(x)

class Discriminator_A(nn.Module):
    training: bool
    
    def setup(self):
        self.net = PatchGANDiscriminator(self.training)
    
    def __call__(self, x):
        return self.net(x)

class Discriminator_B(nn.Module):
    training: bool
    
    def setup(self):
        self.net = PatchGANDiscriminator(self.training)
    
    def __call__(self, x):
        return self.net(x)
    

@jax.vmap
def MAE_loss(y_pred, y_true):
    return (jnp.abs(y_pred-y_true)).sum()

@jax.jit
def loss_g_fn(pm_ga2b, pm_gb2a, 
              pm_da, pm_db, 
              var_ga2b, var_gb2a,
              real_a, real_b):
    
    # This is possible only when Generator_A2B & Generator_B2A has the same structure.
    # If not, you need to define two fns using different apply function.   
    gen_fn = functools.partial(
        Generator_A2B(training=True).apply,
        mutable=["batch_stats"]
    )
    # Same for disc_fn. See above comment.
    disc_fn = functools.partial(
        Discriminator_A(training=True).apply,
    )
    
    # GAN loss for A->B
    tmp = {'params': pm_ga2b, 'batch_stats':var_ga2b["batch_stats"]}
    fake_b, var_ga2b = gen_fn(tmp, real_a)
    
    tmp = {'params':pm_db}
    fake_b_logit = disc_fn(tmp, fake_b)
    loss_gb = -fake_b_logit.mean()
    
    # GAN loss for B->A
    tmp = {'params': pm_gb2a, 'batch_stats':var_gb2a["batch_stats"]}
    fake_a, var_gb2a = gen_fn(tmp, real_b)
    
    tmp = {'params':pm_da}
    fake_a_logit = disc_fn(tmp, fake_a)
    loss_ga = -fake_a_logit.mean()
    
    # Cycle consistency loss
    tmp = {'params': pm_ga2b, 'batch_stats':var_ga2b["batch_stats"]}
    recon_b, var_ga2b = gen_fn(tmp, fake_a)
    loss_gb_cycle = MAE_loss(recon_b, real_b).mean()
    
    tmp = {'params': pm_gb2a, 'batch_stats':var_gb2a["batch_stats"]}
    recon_a, var_gb2a = gen_fn(tmp, fake_b)
    loss_ga_cycle = MAE_loss(recon_a, real_a).mean()
    
    # total loss
    loss_g = loss_ga + loss_gb + 10*(loss_ga_cycle+loss_gb_cycle)
    
    # images
    generated = {
        'fake_a':fake_a,
        'fake_b':fake_b,
        'recon_a':recon_a,
        'recon_b':recon_b
    }
    
    return loss_g, (var_ga2b, var_gb2a, generated)
    
@jax.jit
def loss_d_fn(pm_d, real, fake, gp_rng):
    # For discriminators, we need to use different optimizers.
    # Thus, loss_d_fn for one discriminator is enough, which is simpler than loss_g_fn.
    
    
    # See the comment in loss_g_fn.
    disc_fn = functools.partial(
        Discriminator_A(training=True).apply,
        {'params':pm_d}
    )
    
    fake_logit = disc_fn(fake)
    loss_d_fake = fake_logit.mean()
    
    real_logit = disc_fn(real)
    loss_d_real = real_logit.mean()
    
    # gradient panelty
    alpha = random.uniform(gp_rng, shape=(fake.shape[0],1,1,1), 
                                dtype=jnp.float32, minval=0, maxval=1)
    alpha = jnp.tile(alpha, (1, fake.shape[1], fake.shape[2], fake.shape[3]))
    interpolate = alpha * real + (1-alpha) * fake

    gp = jax.grad(gradient_panelty, argnums=0)(interpolate, pm_d)
    gp = jnp.reshape(gp, (gp.shape[0], -1))
    gp_norm = jnp.sqrt((gp**2).sum(axis=1) + 1e-12)
    gp_norm = ((1-gp_norm)**2).mean()

    loss_d = loss_d_fake - loss_d_real + 10 * gp_norm

    return loss_d

@jax.jit
def gradient_panelty(interpolate, params_d):
    inter_logits = Discriminator_A(training=True).apply({'params':params_d},interpolate)
    return inter_logits.mean()

@jax.jit
def train_step(state_g, state_da, state_db, real_a, real_b, gp_rng):
    # we assume that n_critics=1
    grad_g_fn = jax.value_and_grad(loss_g_fn, argnums=[0,1], has_aux=True)
    grad_d_fn = jax.value_and_grad(loss_d_fn, argnums=0, has_aux=False)

    var_g_a2b = {'batch_stats': state_g.batch_stats_a2b}
    var_g_b2a = {'batch_stats': state_g.batch_stats_b2a}
    
    # update generator
    # aux = (var_ga2b, var_gb2a, var_da, var_db, generated)
    (loss_g, aux), grads_g = grad_g_fn(state_g.params['generator_a2b']['params'],
                                       state_g.params['generator_b2a']['params'],
                                       state_da.params,
                                       state_db.params,
                                       var_g_a2b,
                                       var_g_b2a,
                                       real_a, 
                                       real_b)
    
    # update two models using one optimizer.
    grads_g = {'generator_a2b': {'params':grads_g[0]},
               'generator_b2a': {'params':grads_g[1]}}
    
    state_g = state_g.apply_gradients(grads=grads_g, 
                                      batch_stats_a2b=aux[0]["batch_stats"],
                                      batch_stats_b2a=aux[1]["batch_stats"])
    
    # udpate discriminator
    var_g_a2b = {'batch_stats': state_g.batch_stats_a2b}
    var_g_b2a = {'batch_stats': state_g.batch_stats_b2a}
    
    gp_rng, sub_gp_rng = random.split(gp_rng)
    loss_da, grads_da = grad_d_fn(state_da.params,
                                    real_a,
                                    aux[-1]['fake_a'],
                                    sub_gp_rng)

    state_da = state_da.apply_gradients(grads=grads_da)
    
    gp_rng, sub_gp_rng = random.split(gp_rng)
    loss_db, grads_db = grad_d_fn(state_db.params,
                                    real_b,
                                    aux[-1]['fake_b'],
                                    sub_gp_rng)
    
    state_db = state_db.apply_gradients(grads=grads_db)
    
    return state_g, state_da, state_db, (loss_g, loss_da, loss_db), aux[-1]

def train(num_epochs, train_loader, eval_loader, state_g, state_da, state_db, rng, writer):
    for epoch in range(1, num_epochs+1):
        for i, (real_a, real_b) in enumerate(train_loader):
            
            rng, step_rng = random.split(rng)
            process = lambda in_: in_ * 2. - 1.
            real_a, real_b = list(map(process, [real_a, real_b]))

            state_g, state_da, state_db, losses, generated =\
                    train_step(state_g, state_da, state_db, real_a, real_b, step_rng)

            if (i+1)%50 == 0:
                print(f'Epoch {epoch} | Iteration {i+1} |\
                     Loss_G {losses[0]:.3f} | Loss_DA {losses[1]:.3f} | Loss_DB {losses[2]:.3f}')

                num_steps = len(train_loader) * (epoch-1) + i
                writer.add_scalar('Loss/generator', to_np(losses[0]), num_steps)
                writer.add_scalar('Loss/disc_a', to_np(losses[1]), num_steps)
                writer.add_scalar('Loss/disc_b', to_np(losses[2]), num_steps)
                writer.add_images('A/Real', (to_np(real_a)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('A/Fake', (to_np(generated["fake_b"])+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('A/Recon', (to_np(generated["recon_a"])+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('B/Real', (to_np(real_b)+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('B/Fake', (to_np(generated["fake_a"])+1.)/2., num_steps, dataformats='NHWC')
                writer.add_images('B/Recon', (to_np(generated["recon_b"])+1.)/2., num_steps, dataformats='NHWC')

def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    rng = random.PRNGKey(seed=123)
    
    # create model structures
    generator_a2b = Generator_A2B(training=True)
    generator_b2a = Generator_B2A(training=True)
    discriminator_a = Discriminator_A(training=True)
    discriminator_b = Discriminator_B(training=True)
    
    # initialize model
    dummy_x = jnp.ones((1,128,128,3)) # NCHW
    rng, step_rng = random.split(rng)
    ga2b_state = generator_a2b.init(step_rng, dummy_x)
    rng, step_rng = random.split(rng)
    gb2a_state = generator_b2a.init(step_rng, dummy_x)
    rng, step_rng = random.split(rng)
    da_state = discriminator_a.init(step_rng, dummy_x)
    rng, step_rng = random.split(rng)
    db_state = discriminator_b.init(step_rng, dummy_x)

    # define optimizers
    g_tx = optax.adam(learning_rate=1e-4)
    da_tx = optax.adam(learning_rate=1e-4)
    db_tx = optax.adam(learning_rate=1e-4)

    # create TrainState
    g_params = {'generator_a2b': {'params':ga2b_state['params']},
               'generator_b2a': {'params':gb2a_state['params']}}
    
    # well.. it seems that this is possible only when 
    # generator_a2b and generator_b2a have the same apply method & param/vars.
    # Is there any other way?
    g_train_state = TrainStateCycle.create(
        apply_fn=generator_a2b.apply,
        params=g_params,
        tx=g_tx,
        batch_stats_a2b=ga2b_state['batch_stats'],
        batch_stats_b2a=gb2a_state['batch_stats']
    )

    da_train_state = TrainState.create(
        apply_fn=discriminator_a.apply,
        params=da_state['params'],
        tx=da_tx)
    
    db_train_state = TrainState.create(
        apply_fn=discriminator_b.apply,
        params=db_state['params'],
        tx=db_tx)

    train_loader, test_loader = get_dataloader(batch_size=8)

    save_dir = 'exp_results/generation/wgan_gp_cycle/version_2'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    rng, step_rng = random.split(rng)
    train(200, 
          train_loader, 
          test_loader, 
          g_train_state, 
          da_train_state, 
          db_train_state, 
          step_rng,
          writer)
    

if __name__ == '__main__':
    main()