# type: ignore

import os, glob
from typing import Any
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
from data.utils import get_mnist_dataloader

class TrainStateBN(TrainState):
    batch_stats: Any


class Swish(nn.Module):
    def __call__(self, x):
        return x * jax.nn.sigmoid(x)

class TimeEmbedding(nn.Module):
    n_steps: int
    embed_dim: int
    d_model: int
    
    def setup(self):
        emb = jnp.arange(0, self.d_model, step=2) / self.d_model * jnp.log(10000)
        emb = jnp.exp(-emb)
        pos = jnp.arange(self.n_steps, dtype=jnp.float32)
        emb = pos[:, None] * emb[None, :] # shape = (n_steps, d_model/2)
        emb = jnp.stack([jnp.sin(emb), jnp.cos(emb)], axis=-1) # shape = (n_steps, d_model/2, 2)
        self.embed = jnp.reshape(emb, (self.n_steps, -1)) # shape = (n_steps, d_model)

    @nn.compact
    def __call__(self, t):
        t_emb = self.embed[t]
        t_emb = nn.Dense(features=self.embed_dim)(t_emb)
        t_emb = Swish()(t_emb)
        t_emb = nn.Dense(features=self.embed_dim)(t_emb)
        return t_emb
    
class DownSample(nn.Module):
    # 2x downsample
    out_feats: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Conv(features=self.out_feats, 
                       kernel_size=(3,3), 
                       strides=(2,2), 
                       padding='SAME')(x)
    
class UpSample(nn.Module):
    # 2x upsample
    out_feats: int
    
    @nn.compact
    def __call__(self, x):
        return nn.ConvTranspose(features=self.out_feats,
                                kernel_size=(2, 2),
                                strides=(2,2),
                                padding='SAME')(x)

class AttnBlock(nn.Module):
    feats: int
    
    @nn.compact
    def __call__(self, x):
        h = nn.GroupNorm(32)(x)
        h = nn.SelfAttention(num_heads=1, qkv_features=self.feats)(h)
        h = nn.Conv(self.feats, (1, 1), (1, 1))(h)
        return x + h
    
class ResBlock(nn.Module):
    out_feats: int
    dropout: float
    attention: bool
    training: bool
    
    @nn.compact
    def __call__(self, x, temb):
        # block 1
        h = nn.GroupNorm(32)(x)
        h = Swish()(h)
        h = nn.Conv(self.out_feats, (3,3), (1,1), padding='SAME')(h)
        
        # time embedding projection
        temb = Swish()(temb)
        temb = nn.Dense(features=self.out_feats)(temb) # shape = (B, 1, temb_dim)
        h += temb[:,None,:]
        
        # block 2
        h = nn.GroupNorm(32)(h)
        h = Swish()(h)
        h = nn.Dropout(rate=self.dropout, deterministic=not self.training)(h)
        h = nn.Conv(self.out_feats, (3,3), (1,1))(h)
        
        # short-cut
        if x.shape[-1] != self.out_feats:
            short_cut = nn.Conv(self.out_feats, (1,1), (1,1))(x)
        else:
            short_cut = x
        
        h += short_cut
        
        # attention
        if self.attention:
            h = AttnBlock(self.out_feats)(h)

        return h            
        
class UNet(nn.Module):
    # much more simpler than other framework, since flax automatically set input sizes.
    
    img_ch: int
    ch: int
    ch_mult: list
    n_steps: int
    attn: list
    num_res_blocks: int
    dropout: float
    training: bool
    
    def setup(self):
        temb_dim = self.ch * 4
        self.time_embedding = TimeEmbedding(self.n_steps, embed_dim=temb_dim, d_model=self.ch)
    
    @nn.compact
    def __call__(self, x, t):
        # time embedding
        temb = self.time_embedding(t)
        
        # head
        h = nn.Conv(self.ch, (3,3), (1,1))(x)
        
        # downsampling
        hiddens = [h]
        
        for i, mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                h = ResBlock(out_feats=self.ch*mult, dropout=self.dropout, attention=(i in self.attn), training=self.training)(h, temb)
                hiddens.append(h)
                
            if i != len(self.ch_mult)-1:
                h = DownSample(out_feats=self.ch*mult)(h)
                hiddens.append(h)
                    
        # middle
        h = ResBlock(out_feats=self.ch*self.ch_mult[-1], dropout=self.dropout, attention=True, training=self.training)(h, temb)
        h = ResBlock(out_feats=self.ch*self.ch_mult[-1], dropout=self.dropout, attention=False, training=self.training)(h, temb)
        
        # upsampling
        for i, mult in enumerate(self.ch_mult[::-1]):
            for _ in range(self.num_res_blocks + 1):
                print(h.shape, hiddens[-1].shape)
                h = jnp.concatenate([h, hiddens.pop()], axis=-1)
                h = ResBlock(out_feats=self.ch*mult, dropout=self.dropout, attention=(i in self.attn), training=self.training)(h, temb)
                
            if i != 0:
                h = UpSample(out_feats=self.ch*mult)(h)
            
        # tail
        h = nn.GroupNorm(32)(h)
        h = Swish()(h)
        h = nn.Conv(img_ch, (3,3), (1,1))(h)
        
        return h


class DDPM(nn.Module):
    training: bool
    n_steps: int
    beta_0: float
    beta_T: float
    
    def setup(self):
        self.net = Unet(self.training)
        self.betas = jnp.linspace(beta_0, beta_T, n_steps)
        alpha_bar = jnp.cumprod(1.0 - self.betas)
        self.sqrt_alpha_bar = jnp.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - alpha_bar)
    
    def sample(self, noise):
        pass
    
    def __call__(self, rngs, x, t):
        noise = jax.random.normal(rngs, shape=x.shape)
        return None        


@jax.jit
def train_step(state:TrainState, x:jnp.array):
    def loss_fn(params):
        pass
    
    return None

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
    
    rngs = random.PRNGKey(seed=123)
    init_rngs = {'params': random.PRNGKey(seed=0), 'dropout': random.PRNGKey(seed=1)}
    ddpm = UNet(3, 64, [2,4,6], 100, [1], 2, 0.2, True)
    state = ddpm.init(init_rngs, jnp.ones((10, 32, 32, 1)), jnp.ones((10,1), dtype=jnp.int32))
    
    tx = optax.adam(learning_rate=1e-3)
    train_state = TrainSTate.create(apply_fn=ddpm.apply,
                                    params=state["params"],
                                    tx=tx)
    
    train_loader, test_loader = get_mnist_dataloader(batch_size=8)
    
    save_dir = 'exp_results/MNIST/ddpm/'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    train(1, train_loader, train_state, writer)


if __name__ == '__main__':
    main()