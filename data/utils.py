import os
import numpy as np
import jax.numpy as jnp
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Compose
from PIL.Image import BICUBIC
from PIL import Image

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

class NumpyCast(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)/255.
    
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_flat_mnist_dataloader(batch_size):
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

def get_mnist_dataloader(batch_size):
    os.makedirs('data/', exist_ok=True)
    transform = Compose([Resize(size=(32, 32)), NumpyCast()])

    train_ds = MNIST('data/', train=True, download=True, transform=transform)
    test_ds = MNIST('data/', train=False, download=True, transform=transform)
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

class MNISTDenoiseDataset(MNIST):
    def __getitem__(self, index: int):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # add noise . assume img scale = [0, 1]
        noisy = img + np.random.normal(0, scale=1.0, size=img.shape)
        noisy = np.clip(noisy, img.min(), img.max())

        return noisy, img
    
def get_denoise_mnist_dataloader(batch_size):
    os.makedirs('data/', exist_ok=True)
    transform = Compose([Resize(size=(32, 32)), NumpyCast()])

    train_ds = MNISTDenoiseDataset('data/', train=True, download=True, transform=transform)
    test_ds = MNISTDenoiseDataset('data/', train=False, download=True, transform=transform)
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