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

class DCGAN(nn.Module):
    pass


def main():
    pass

if __name__ == '__main__':
    main()