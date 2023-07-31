import os
import orbax.checkpoint
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union, Type, List
from jax import numpy as jnp, random
from flax import linen as nn
from tqdm import tqdm

def load_checkpoint(ckpt_dir: str,
                    ckpt_file: str,
                    model: Type[nn.Module]) -> Any:
    """
    Load checkpoint from local directory
    
    Args:
    -----
        ckpt_dir: str
            Local directory where checkpoint file is saved.
        ckpt_file: str
            Name of checkpoint file.
        model: nn.Module
            Model for which the checkpoint is being loaded
        
    Returns:
    -------
        Any: Model with trained checkpoint parameters
    """
    # Get absolute path to checkpoint directory
    ckpt_dir = os.path.join(os.path.expanduser('~'), str(ckpt_dir))
    
    # Check if checkpoint file exists in locoal directory
    if not os.path.exists(os.path.join(ckpt_dir, ckpt_file)):
        raise FileNotFoundError(f"No checkpoints file found at {os.path.join(ckpt_dir, ckpt_file)}")
        

    # Create a Checkpointer using the PyTreeCheckpointHandler
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    
    # Restore state dictionary from the checkpoint path
    state_dict = ckptr.restore(ckpt_dir, item=None)
    
    # Update model parameters using restored state dict
    params = {'params': state_dict['params']}
    return model.bind(params)

def sample_norm(N, D, size, key):
    """
    Sample noise from perturbation kernel p_r.
    Sampling from inverse-beta distribution found in Appendix B.
    """
    R1 = random.beta(a=N / 2., b=D / 2., key=key, shape=(size,))
    R2 = R1 / (1 - R1 +1e-8)
    return R2

def sample_loop(updated_model: nn.Module, 
                num_steps: int = 50, 
                max_t: float = 80, 
                min_t: float = 0.1, 
                sample_size: int = 200, 
                S_churn: float = 0, 
                S_min: float = 0.01, 
                S_max: float = 5, 
                S_noise: float = 1, 
                D: int = 2048, 
                N: int = 2, 
                rho: int = 7, 
                std_data: float = 0.5, 
                key_seed: int = 32) -> jnp.array:
    """
    Performs the sampling loop for a diffusion model.

    Args:
        updated_model: The PFGM++ model with parameters loaded from a checkpoint.
        num_steps: Number of steps in the sampling loop.
        max_t: Maximum time step in the discretization.
        min_t: Minimum time step in the discretization.
        sample_size: Number of samples to be generated.
        S_churn: Parameter for temporary noise increase.
        S_min: Minimum limit for gamma.
        S_max: Maximum limit for gamma.
        S_noise: Parameter for noise level in sampling.
        D: Parameter related to the dimensionality of the data.
        N: Parameter related to the dimensionality of the data.
        rho: Parameter for time step discretization.
        std_data: Standard deviation of the data.
        key: Random number generator key.

    Returns:
        jnp.array: The resulting samples after the sampling loop.
    """
    # Split the keys, so we have different RNGs in the sampling process 
    # ^ DOES THIS MAKE SENSE ^^^
    key = random.PRNGKey(key_seed)
    rng, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
    
    # Latent space vectors
    latents = random.normal(subkey1, shape=(sample_size, N)) 
    
    # Time step discretization (t = sigma ~ r)
    step_indices = jnp.arange(num_steps)
    t_steps = (max_t**(1/rho) + step_indices / (num_steps-1) * (min_t**(1/rho) - max_t**(1/rho)))**rho
    t_steps = jnp.concatenate([t_steps, jnp.zeros_like(t_steps[:1])]) # Convert t_steps to arr & add zero to end i.e. t_N=0

    r_max = t_steps[0] * jnp.sqrt(D) # r=sigma\sqrt{D} formula

    # Sample noise from perturbation kernel p_r(R) = sampling from inverse-beta
    # by change-of-variable (Appendix B of PFGM++)
    R2 = sample_norm(N, D, sample_size, subkey2) 
    R3 = (r_max.squeeze() * jnp.sqrt(R2 + 1e-8)).reshape(len(R2), -1)

    # Uniformly sample the angle component 
    gaussian = random.normal(subkey3, shape=(sample_size, N))
    unit_gaussian = gaussian / jnp.linalg.norm(gaussian, ord=2, axis=1, keepdims=True)

    # Construct the perturbation (similar to Algo 2. in Karras et al. 2022)
    x_next = unit_gaussian * R3
    x_orig = x_next

    # Loop over consecutive elements in 't_steps', while keeping
    # track of the index of each pair
    for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
        x_cur = x_next

        # Increase noise temporarily
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + jnp.sqrt(t_hat**2 - t_cur**2) * S_noise * random.normal(subkey4, shape=x_cur.shape)

        # Euler step & preconditions (Table 1 in Karras et al. 2022)
        c_in = 1 / jnp.sqrt(std_data**2 + t_hat**2)
        denoised = updated_model(x_hat * c_in, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction
        if i < num_steps - 1:
            # Preconditions
            c_in = 1 / jnp.sqrt(std_data**2 + t_next**2)
            denoised = updated_model(x_next * c_in, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next, x_orig