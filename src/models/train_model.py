import jax
from jax import jit
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import beta
import flax
from flax import jax_utils
from flax import linen as nn
import optax
import orbax.checkpoint
from flax.training import train_state, orbax_utils
import numpy as np
import os
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union, Type, List
import wandb
from tqdm import tqdm

def init_train_state(model: Any,
                     random_key: Any,
                     x_shape: tuple,
                     t_shape: tuple,
                     learning_rate: int) -> train_state.TrainState:
    """
    Function to initialize the TrainState dataclass, whcih represents
    the entire training state, including step number, parameteers, and 
    optimizer state. Used in a Flax framework because you no longer need
    to initialize the model again & again with new variables. Rather we 
    just update the "state" of the model and pass this as inputs to functions.
    
    Args:
    -----
        model: nn.Module
            Model we want to train.
        random_key: jax.random.PRNGKey()
            Used to trigger the initialization functions, which generate
            the initial set of parameters that the model will use.
        x_shape: tuple
            Shape of the batch of data (x) that will be input into the model.
            Used to trigger shape inference.
        t_shape: tuple
            Shape of the batch of data (t) that will be input into the model.
            Used to trigger shape inference.
        learning_rate: float
            How large of a step the optimizer should take.
            
    Returns:
    --------
        train_state.TrainState:
            A utility class for handling parameter and gradient updates.
    """
    # Initialize the model
    variables = model.init(random_key, jnp.ones(x_shape), jnp.ones(t_shape))
    
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    
    # Create a state
    return flax.training.train_state.TrainState.create(apply_fn=model.apply,
                                                       tx=optimizer,
                                                       params=variables['params'])

def sample_norm(N, D, size, key):
    """
    Sample noise from perturbation kernel p_r.
    Sampling from inverse-beta distribution found in Appendix B.
    """
    R1 = random.beta(a=N / 2., b=D / 2., key=key, shape=(size,))
    R2 = R1 / (1 - R1 +1e-8)
    return R2

def train_step(state, batch, std_data, D, N, key_seed):
    """
    Args:
    ----
        D: int
            The extra dimensions (D>1)
        N: int
            The dimensionality of the data
    """
    # Define the loss function
    def loss_fn(params):
        key = random.PRNGKey(key_seed)
        
        # Sampling from the noise distribution (Table 1 Karras et al. 2022)
        rnd_normal = random.normal(key, shape=(batch.shape[0], 1))
        t = jnp.exp(rnd_normal * 1.2 - 1.2)
        # Phase alignment of Diffusion Models -> PFGM++ (Appendix C)
        # Note: t = sigma in the "EDM" design choice (Karras et al. 2022)
        r = t * jnp.sqrt(D)
        
        # Sampling from p_r(R) by change-of-variable (Appendix B)
        R2 = sample_norm(N, D, batch.shape[0], key)
        R3 = (r.squeeze() * jnp.sqrt(R2 +1e-8)).reshape(len(R2), -1)
        
        # Sampling the uniform angle component
        unit_gaussian = random.normal(key, shape=(batch.shape[0], N))
        unit_gaussian = unit_gaussian / jnp.linalg.norm(unit_gaussian, ord=2, axis=1, keepdims=True)
        x_hat = batch + (unit_gaussian * R3)
        
        # Preconditions (Table 1 & Sect. 5 of Karras et al. 2022)
        c_in = 1 / jnp.sqrt(std_data**2 + t**2)
        c_out = t * std_data / jnp.sqrt(std_data**2 + t**2)
        c_skip = std_data**2 / (std_data**2 + t**2)

        # f_theta in Karras et al. 2022
        D_x = state.apply_fn({'params': params}, x_hat * c_in, t)
        
        # lambda(theta) in Karras et al. 2022 
        # using the loss weighting of "EDM"
        weight = (t**2 + std_data ** 2) / (t*std_data) ** 2
        # Calculate weighted loss
        loss = jnp.sum(weight * ((D_x - batch) ** 2), axis=1).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)
    return new_state, loss

def accumulate_metrics(metrics):
    """
    Function that accumulates all the per batch metrics into
    per epoch metrics
    """
    metrics = jax.device_get(metrics)
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }


def save_checkpoint(ckpt_dir: str,
                    state: Any,
                    step: int,
                    wandb_logging: bool = False) -> None:
    """
    Save the training state as a checkpoint

    Args:
    -----
        ckpt_dir: str
            Directory to save the checkpoint files.
        state: Any
            The training state to be saved.
        step: int
            Current training step or epoch number.
        wandb_logging: bool
            If True, uses the wandb run name in the checkpoint filename.
            Default is False.
    """
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())

    # Ensure directory exists
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get the wandb run name or id if wandb logging is enabled
    run_name = wandb.run.name if wandb_logging else ""

    # Create checkpoint file path with the ".flax" extension and the wandb run name if applicable
    ckpt_file = os.path.join(ckpt_dir, f"checkpoint_{run_name}.flax")

    # Save checkpoint to local directory
    ckptr.save(ckpt_file, state,
               save_args=flax.training.orbax_utils.save_args_from_target(state),
               force=True)

    # If wandb logging is enabled, save the checkpoint to wandb run
    if wandb_logging:
        wandb.save(ckpt_file)

def train_model(train_loader, model, state, config, rng_seed=21, wandb_logging=False, project_name='toy_pfgmpp', job_type='simple_noise_net', dir_name='PFGMPP/saved_models/toy'):
    """
    Train a machine learning model with optional Weights & Biases (wandb) logging.

    Parameters:
    -----------
    train_loader: 
        A data loader providing the training data.
    model: 
        The model to be trained.
    state: 
        The initial state of the model.
    config: dict
        A dictionary containing configuration parameters for training, such as learning rate, batch size etc.
    rng_seed: int
        PRNG seed
    wandb_logging: bool
        If True, training progress is logged using wandb.
        Default is False.
    project_name: str
        The name of the wandb project. Only used if wandb_logging is True.
        Default is 'toy_pfgmpp'.
    job_type: str
        The type of job for wandb logging. Only used if wandb_logging is True.
        Default is 'simple_noise_net'.
    dir_name: str
        The directory where the model checkpoints will be saved. Also used for wandb logging if wandb_logging is True. Default is 'saved_models/toy'.

    Returns:
    --------
        model: The trained model.
        state: The final state of the model after training.
    """
    # Get the absolute path of the saved_models/toy directory
    dir_name = os.path.join(os.path.expanduser('~'), str(dir_name))
    # If wandb logging is enabled, initialize wandb
    if wandb_logging:
        wandb.init(project=project_name, job_type=job_type, dir= dir_name)
        wandb.config.update(config)

    # Start the training loop
    rng = random.PRNGKey(rng_seed)
    for epoch in tqdm(range(config['epochs'])):
        rng, input_rng = jax.random.split(rng)
        # Initialize a list to store all batch-level metrics
        batch_metrics = []

        for batch in train_loader:
            # Prepare the data
            batch = jax.device_put(batch)

            # Update the model
            train_step_jit = jax.jit(train_step, static_argnums=(3,4,5))
            state, batch_loss = train_step_jit(state, batch, config['std_data'], config['D'], config['N'], config['seed'])    # UPDATE THIS 

            # Store the batch-level metric in the list
            batch_metrics.append({'Train Loss': batch_loss})

        # Use accumulate_metrics to calculate average metrics for the epoch
        epoch_metrics = accumulate_metrics(batch_metrics)

        # If wandb logging is enabled, log metrics
        if wandb_logging:
            wandb.log(epoch_metrics)
        
    # Save checkpoint
    checkpt_dir = dir_name # dir to save the checkpoints
    save_checkpoint(checkpt_dir, state, config['epochs'], project_name)

    # # If wandb logging is enabled, save the model checkpoint
    # if wandb_logging:
    #     base_path = os.path.dirname(os.path.abspath(dir_name))
    #     wandb.save(os.path.join(checkpt_dir, f"checkpoint_{config['epochs']}.flax"), base_path=base_path)

    return model, state


def train_model_sweep(train_loader, model, state, config, key_seed=47, wandb_logging=True):
    """
    Train a machine learning model with optional Weights & Biases (wandb) logging.

    Parameters:
    -----------
    train_loader: 
        A data loader providing the training data.
    model: 
        The model to be trained.
    state: 
        The initial state of the model.
    config: dict
        A dictionary containing configuration parameters for training, such as learning rate, batch size etc.
    wandb_logging: bool
        If True, training progress is logged using wandb.
        Default is False.

    Returns:
    --------
        model: The trained model.
        state: The final state of the model after training.
    """
    # Initialize a var to hold the best test loss seen so far
    best_train_loss = float('inf')
    
    # Start the training loop
    for epoch in tqdm(range(config['epochs'])):
        # Initialize a list to store all batch-level metrics
        batch_metrics = []

        for batch in train_loader:
            # Prepare the data
            batch = jax.device_put(batch)

            # Update the model
            train_step_jit = jax.jit(train_step, static_argnums=(3,4,5))
            state, batch_loss = train_step_jit(state, batch, config['std_data'], config['D'], config['N'], config['train_model_key'])    

            # Store the batch-level metric in the list
            batch_metrics.append({'Train Loss': batch_loss})

        # Use accumulate_metrics to calculate average metrics for the epoch
        epoch_metrics = accumulate_metrics(batch_metrics)

        # If the train loss for this epoch is better than the previous best,
        # save the model
        if epoch_metrics['Train Loss'] < best_train_loss:
            best_train_loss = epoch_metrics['Train Loss'] # Update the best train loss
            checkpt_dir = dir_name # dir where models are saved to
            save_checkpoint(checkpt_dir, state, epoch, project_name)

        
        # If wandb logging is enabled, log metrics
        if wandb_logging:
            wandb.log(epoch_metrics)

    return model, state
