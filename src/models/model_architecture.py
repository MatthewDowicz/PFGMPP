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






class DoubleDense(nn.Module):
    features: int

    def setup(self):
        self.dense1 = nn.Dense(self.features)
        self.act1 = nn.relu
        self.dense2 = nn.Dense(self.features)
        self.act2 = nn.relu

    def __call__(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x

class Down(nn.Module):
    features: int

    def setup(self):
        self.dense1 = DoubleDense(self.features)
        self.emb_layer = nn.Dense(self.features)

    def __call__(self, x, t):
        x = self.dense1(x)
        emb = self.emb_layer(t)
        return x + emb


class Up(nn.Module):
    features: int

    def setup(self):
        self.dense1 = DoubleDense(self.features)
        self.emb_layer = nn.Dense(self.features)

    def __call__(self, x, skip_x, t):
        x = jnp.concatenate([skip_x, x], axis=-1)
        x = self.dense1(x)
        emb = self.emb_layer(t)
        return x + emb


class UNet(nn.Module):
    depth: int = 4
    initial_feature: int = 64
    out_feature: int = 2
    std_data: float = 0.5
    embedding_dim: int = 64

    def setup(self):
        # Initial dense layer
        self.inc = DoubleDense(self.initial_feature) # 2->64

        # Encoder Block (Downsampling)
        for i in range(1, self.depth):
            features = self.initial_feature * (2 ** i)
            # print('Encoder features', features)
            setattr(self, f'down{i}', Down(features))

        # Bottleneck Layers
        bottleneck_features = self.initial_feature * (2 ** (self.depth-1))
        self.bot1 = DoubleDense(bottleneck_features) # 512->512
        self.bot2 = DoubleDense(bottleneck_features * 2) # 512->1024
        self.bot3 = DoubleDense(bottleneck_features) # 1024->512

        # Decoder Block (Upsampling)
        for i in reversed(range(0, self.depth-1)):
            features = self.initial_feature * (2 ** i)
            setattr(self, f'up{i}', Up(features))

        # Final Output Layer
        self.outc = nn.Dense(self.out_feature)

    def __call__(self, x, t):
        # Preconditioning terms
        t = t.squeeze()
        c_out = t * self.std_data / jnp.sqrt(self.std_data**2 + t**2)
        c_skip = self.std_data**2 / (self.std_data**2 + t**2)

        # Sampling the noise and embedding the noise via positional encoding
        t = jnp.log(t.flatten()) / 4.
        t = self.pos_encoding(t, self.embedding_dim)
        x_orig = x

        skip_connections = []

        # Pass through the initial layer
        x = self.inc(x) # 2 -> 64
        skip_connections.append(x) # Store the output for skip connection

        # Pass through the dynamic encoder layers
        for i in range(1, self.depth):
            x = getattr(self, f'down{i}')(x, t)
            skip_connections.append(x) # Store the outputs for skip connections
            
        # Pass through the bottleneck layers
        x = self.bot1(x) # 256 -> 512
        x = self.bot2(x) # 512 -> 1024
        x = self.bot3(x) # 1024 -> 512

        # Pass through the dynamic decoder (upsampling) layers
        skip_connections.pop() # THIS IS FOR TESTING PURPOSES FOUND THERE WAS A DISCREPANCY
                       # WITH THE NUMBER OF PARAMS BETWEEN STATIC AND DYNAMIC MODEL
                       # DUE TO THE SKIP CONNECTIONS TAKING WRONG DIMENSIONAL DATA
        for i in reversed(range(0, self.depth-1)):
            skip_output = skip_connections.pop() # Retrieve last stored output
            x = getattr(self, f'up{i}')(x, skip_output, t)

        # Pass through the final output layer
        output = self.outc(x) # 64 -> 2

        # Reshape c_out & c_skip to match dimensions for broadcasting
        c_out = jnp.reshape(c_out, (-1,1))
        c_skip = jnp.reshape(c_skip, (-1,1))
        return c_out * output + c_skip * x_orig

    def pos_encoding(self, t, channels):
        t = jnp.expand_dims(t, axis=-1)  # Add an additional dimension to t
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, channels, 2).astype(jnp.float32) / channels))
        pos_enc_a = jnp.sin(t * inv_freq)
        pos_enc_b = jnp.cos(t * inv_freq)
        pos_enc = jnp.concatenate([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc