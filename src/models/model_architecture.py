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






class DoubleConvolution(nn.Module):
    filters: int
    kernel_size: int = (3, 3)
    strides: int = (1, 1)
    padding: int = "SAME"

    def setup(self):
        self.conv1 = nn.Conv(self.filters, self.kernel_size,
                             self.strides, padding=self.padding)
        self.act1 = nn.relu

        self.conv2 = nn.Conv(self.filters, self.kernel_size,
                             self.strides, padding=self.padding)
        self.act2 = nn.relu

    def __call__(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class DownSample(nn.Module):
    window_shape: int = (2, 2)
    strides: int = (2, 2)
    padding: str = "VALID"

    def __call__(self, x):
        return nn.max_pool(x, window_shape=self.window_shape,
                           strides=self.strides, padding=self.padding)

class UpSample(nn.Module):
    filters: int
    kernel_size: int = (2, 2)
    strides: int = (2, 2)
    padding: int = (1, 1)

    def setup(self):
        self.up_conv = nn.ConvTranspose(self.filters,
                                       self.kernel_size,
                                       strides=self.strides,
                                       padding=self.padding)

    def __call__(self, x):
        return self.up_conv(x)

class UNet(nn.Module):
    depth: int = 4
    initial_filters: int = 64
    output_channels: int = 3

    def setup(self):
        # Encoder
        self.down_blocks = [DoubleConvolution(self.initial_filters * 2**i) for i in range(self.depth)]
        self.downsamples = [DownSample() for _ in range(self.depth)]
        
        # Bottleneck 
        self.bottleneck_block = DoubleConvolution(self.initial_filters * 2**self.depth)

        # Decoder
        self.up_samples = [UpSample(self.initial_filters * 2**(i-1)) for i in range(self.depth, 0, -1)]
        self.up_blocks = [DoubleConvolution(self.initial_filters * 2**i) for i in range(self.depth-1, -1, -1)]

        # Final Convolutional Layer
        self.final_conv = nn.Conv(self.output_channels, 
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding="SAME")

    def __call__(self, x):
        skip_connections = []
        
        # Encoder path
        for i in range(self.depth):
            x = self.down_blocks[i](x)
            skip_connections.append(x)
            x = self.downsamples[i](x)
            # print(f'Encoder{i+1} x.shape =', x.shape)

        # Bottleneck
        x = self.bottleneck_block(x)
        # print('Bottleneck x.shape =', x.shape)

        # Decoder path
        for i in range(self.depth):
            x = self.up_samples[i](x)
            x = jnp.concatenate([x, skip_connections.pop()], axis=-1)
            # print(f'Skip_connection{i+1} x.shape =', x.shape)
            x = self.up_blocks[i](x)
            # print(f'Decoder{i+1} x.shape =', x.shape)

        # Final Convolution layer
        x = self.final_conv(x)
        # print('Final x.shape =', x.shape)

        return x
