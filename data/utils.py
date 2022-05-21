import os
import numpy as np
import jax.numpy as jnp
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_mnist_dataloader(batch_size):
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

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))