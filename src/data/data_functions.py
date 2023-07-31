import torch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

def generate_data(key_seed, size=4000, std=0.2, mean_scale=1):
    key = random.PRNGKey(key_seed)
    rng, subkey1, subkey2 = random.split(key, num=3)

    # Generate data for X_0 & X_1
    X_0 = mean_scale + random.normal(subkey1, shape=(size, 2)) * std
    X_1 = -mean_scale + random.normal(subkey2, shape=(size, 2)) * std

    # Concatenate
    X = jnp.concatenate([X_0, X_1], axis=0)
    
    return X


class JaxDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return np.array(self.X[idx])

def numpy_collate(batch):
    """
    Function to allow jnp.arrays to be used in PyTorch Dataloaders.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
class NumpyLoader(torch.utils.data.DataLoader):
    """
    Custom PyTorch DataLoader for numpy/JAX arrays
    """
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)