import os
import numpy as np
import jax.numpy as jnp
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Compose, RandomResizedCrop
from PIL.Image import BICUBIC
from PIL import Image
import random

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

class UnalignedDataset(VisionDataset):
    def __init__(self, root_dir, phase, transform=None):
        
        self.dir_A = os.path.join(root_dir, phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(root_dir, phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted([os.path.join(self.dir_A,x) for x in os.listdir(self.dir_A)])   # load images from '/path/to/data/trainA'
        self.B_paths = sorted([os.path.join(self.dir_B,x) for x in os.listdir(self.dir_B)])    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = transform
        self.transform_B = transform

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        if self.transform_A is not None:
            A_img = self.transform_A(A_img)
        if self.transform_B is not None:
            B_img = self.transform_B(B_img)

        return A_img, B_img

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

def get_horse2zebra_dataloader(batch_size):
    transform = Compose([RandomResizedCrop((128, 128)), NumpyCast()])

    train_ds = UnalignedDataset('data/horse2zebra/', phase="train", transform=transform)
    test_ds = UnalignedDataset('data/horse2zebra/', phase="test", transform=transform)
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