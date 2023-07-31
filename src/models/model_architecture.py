import jax
import jax.numpy as jnp
from jax import jit
from jax import random
import flax
from flax import linen as nn
import numpy as np



def getPositionEncoding(t: jnp.array, d: int, n: int = 10000) -> np.ndarray:
    seq_len = len(t)

    # Calculate the denominator for all dimensions
    denominator = n ** (2.0 * (jnp.arange(d) // 2) / d)

    # Calculate the encoding for all positions and dimensions
    t = t[:, None]  # make it a column vector
    t_scaled = t / denominator  # this will broadcast t along the columns

    # Generate positional encodings
    P = jnp.zeros((seq_len, d))
    sin_component = jnp.sin(t_scaled[:, :d//2])
    cos_component = jnp.cos(t_scaled[:, :d//2])
    
    P = P.at[:, 0::2].set(sin_component)
    P = P.at[:, 1::2].set(cos_component)

    return P

class ScoreNet(nn.Module):
    """
    Common architecture for score-based generative models,
    """
    dim: int = 10  
    latent_dim: int = 32
    std_data: float = 0.5
    
    @nn.compact
    def __call__(self, x, t):
        # Preconditioning terms in the "EDM" model
        t = t.squeeze()  # Ensure t is scalar if it's not
        c_out = t * self.std_data / jnp.sqrt(self.std_data**2 + t**2)
        c_skip = self.std_data**2 / (self.std_data**2 + t**2)

        t = jnp.log(t.flatten()) / 4.
        
        # Basic Network architecture
        x_1 = nn.Dense(self.latent_dim)(x) + getPositionEncoding(t, d=self.latent_dim)
        x_2 = nn.relu(x_1)
        x_3 = nn.relu(nn.Dense(self.latent_dim)(x_2)) + nn.Dense(self.latent_dim)(x_1) 
        x_4 = nn.relu(nn.Dense(self.latent_dim)(x_3)) + nn.Dense(self.latent_dim)(x)
        x_5 = nn.relu(nn.Dense(self.latent_dim)(x_4)) 
        x_6 = nn.relu(nn.Dense(self.latent_dim)(x_5)) 
        x_7 = nn.Dense(self.dim)(x_6)
                
        # Reshape c_out and c_skip to match dimensions for broadcasting
        c_out = jnp.reshape(c_out, (-1, 1))
        c_skip = jnp.reshape(c_skip, (-1, 1))
        
        return c_out * x_7 + c_skip * x