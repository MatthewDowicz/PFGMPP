# Basic Imports
import numpy as np
import matplotlib.pyplot as plt

# Changing fonts to be latex typesetting
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif'


def sampling2D(prior_data, true_data, generated_data, std=0.2, mean_scale=1):
    var_data = std**2 + mean_scale**2
    std_data = np.sqrt(var_data)

    generated_data = prior_data * std_data

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4))
    
    ax0.scatter(prior_data[:, 0], prior_data[:, 1], label='Generated Data', alpha=0.5)
    # ax0.set_title(f'$D={D}$', fontsize=20)
    ax0.set_xlabel('$x_0$', fontsize=15)
    ax0.set_ylabel('$x_1$', fontsize=15)
    ax0.legend()
    
    ax1.scatter(generated_data[:, 0], generated_data[:, 1], label='Denoised Generated data', alpha=0.5)
    # ax1.set_title(f'$D={D}$', fontsize=20)
    ax1.set_xlabel('$x_0$', fontsize=15)
    ax1.set_ylabel('$x_1$', fontsize=15)
    ax1.legend()
    
    ax2.scatter(true_data[:, 0], true_data[:, 1], label='True Data', alpha=0.5, color='orange')
    # ax2.set_title(f'$D={D}$', fontsize=20)
    ax2.set_xlabel('$x_0$', fontsize=15)
    ax2.set_ylabel('$x_1$', fontsize=15)
    ax2.legend()