# Misc utility modules

import torch
from torch import nn
import matplotlib.pyplot as plt


def init_random_weights(P: torch.Tensor):
    """Initialize random weights that respect the network topology
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    """

    n = P.shape[0]
    weights = torch.distributions.Uniform(0, 0.1).sample((n, n))
    weights = torch.where(P > 0, weights, 0)

    return weights


class ZeroClipper(object):
    """Clip the weights to zero if they are negative"""

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        
        if hasattr(module, "A"):
            w = module.A.data
            nn.ReLU(inplace=True)(w)


def plot_losses(losses, title, xlabel, ylabel, outfile):
    """
        Plots loss from training process
    """

    plt.plot(losses, marker="o", color="blue", markersize=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.show()
    plt.savefig(outfile, dpi=300)




