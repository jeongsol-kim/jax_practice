import os, glob
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from data.utils import get_mnist_dataloader

class ResidualBlock(nn.Module):
    num_hiddens: int
    num_residual_hiddens: int
    
    @nn.compact
    def __call__(self, x):
        tmp = nn.relu(x)
        tmp = nn.Conv(features=self.num_residual_hiddens,
                      kernel_size=(3,3),
                      strides=(1,1),
                      padding='SAME',
                      use_bias=False)(tmp)
        tmp = nn.relu(tmp)
        tmp = nn.Conv(features=self.num_hiddens,
                      kernel_size=(1,1),
                      strides=(1,1),
                      use_bias=False)(tmp)
        return tmp + x
    
class ResidualStack(nn.Module):
    num_hiddens: int
    num_residual_hiddens: int
    num_residual_layers: int
    
    def setup(self):
        # use this instead of "append"
        self.layers = [ResidualBlock(self.num_hiddens, self.num_residual_hiddens) \
                       for _ in range(self.num_residual_layers)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return nn.relu(x)
    
class Encoder(nn.Module):
    num_hiddens: int
    num_residual_layers: int
    num_residual_hiddens: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.num_hiddens//2,
                    kernel_size=(4,4),
                    strides=(2,2),
                    padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_hiddens,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_hiddens,
                    kernel_size=(3,3),
                    strides=(1, 1),
                    padding='SAME')(x)
        x = ResidualStack(self.num_hiddens,
                          self.num_residual_layers,
                          self.num_residual_hiddens)(x)
        return x

class Decoder(nn.Module):
    num_hiddens: int
    num_residual_layers: int
    num_residual_hiddens: int
    out_channels: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.num_hiddens,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='SAME')(x)
        x = ResidualStack(self.num_hiddens,
                          self.num_residual_layers,
                          self.num_residual_hiddens)(x)
        x = nn.ConvTranspose(features=self.num_hiddens//2,
                             kernel_size=(4,4),
                             strides=(2,2),
                             padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.out_channels,
                             kernel_size=(4,4),
                             strides=(2,2),
                             padding='SAME')(x)
        return x
        

def to_np(a):
    return np.asarray(a)

class TrainStateVQVAE(TrainState):
    commitment_cost: float

class VQVAE(nn.Module):
    num_embeddings: int
    embedding_dim: int
    input_channels: int
    num_hiddens: int
    num_residual_layers: int
    num_residual_hiddens: int
    
    def setup(self):
        embedding_init = jax.nn.initializers.normal(stddev=1.0)
        self.embedding = nn.Embed(self.num_embeddings,
                                  self.embedding_dim,
                                  embedding_init=embedding_init)
        
        self.encoder = Encoder(self.num_hiddens,
                               self.num_residual_layers,
                               self.num_residual_hiddens)
        
        self.decoder = Decoder(self.num_hiddens,
                               self.num_residual_layers,
                               self.num_residual_hiddens,
                               self.input_channels)
    
    def __call__(self, x):
        # encode
        feats = self.encoder(x) # BxHxWxC
        flat_feats = jnp.reshape(feats, (-1, self.embedding_dim))
        
        # quantization
        distance = (flat_feats**2).sum(axis=1, keepdims=True) \
                    + jnp.transpose((self.embedding.embedding**2).sum(axis=1, keepdims=True)) \
                    - 2*jnp.dot(flat_feats, jnp.transpose(self.embedding.embedding))
        
        encode_idx = jnp.argmin(distance, axis=1)
        quantized = self.embedding(encode_idx)
        quantized = jnp.reshape(quantized, feats.shape)
        
        # decode
        recon = self.decoder(quantized)
        return recon, feats, quantized

@jax.vmap
def MSE_loss(y_pred, y_true):
    return ((y_pred-y_true)**2).sum()**0.5

@jax.vmap
def latent_loss(quantized, input_feats, commitment_cost):
    e_latent_loss = MSE_loss(jax.lax.stop_gradient(quantized), input_feats)
    q_latent_loss = MSE_loss(quantized, jax.lax.stop_gradient(input_feats))    
    loss = q_latent_loss + commitment_cost * e_latent_loss
    return loss

@jax.jit
def train_step(state:TrainState, x:jnp.array):
    def loss_fn(params):
        commitment_cost = jnp.array([state.commitment_cost]*x.shape[0])
        recon, input_feats, quantized = state.apply_fn({'params': params}, x)
        recon_loss = MSE_loss(recon, x).mean()
        vq_loss = latent_loss(quantized, input_feats, commitment_cost).mean()
        return vq_loss, (recon)
    
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, (recon)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
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
    
    rngs = random.PRNGKey(seed=123)
    vqvae = VQVAE(num_embeddings=512, embedding_dim=128, 
                  input_channels=1,
                  num_hiddens=128, num_residual_layers=2,
                  num_residual_hiddens=32)
    
    state = vqvae.init(rngs, jnp.ones((10, 32, 32, 1)))
    tx = optax.adam(learning_rate=1e-3)
    train_state = TrainStateVQVAE.create(apply_fn=vqvae.apply,
                                    params=state["params"],
                                    tx=tx,
                                    commitment_cost=.25)
    
    train_loader, test_loader = get_mnist_dataloader(batch_size=8)
    
    save_dir = 'exp_results/MNIST/vqvae/'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    train(1, train_loader, train_state, writer)

if __name__=='__main__':
    main()