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


def init_random_weights_priv_noise(
    P: torch.Tensor, E: torch.Tensor, D: torch.Tensor, R: torch.float
):
    """Initialize random weights that respect the network topology and feasible privacy noise
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param E: epsilon values for peer-to-peer privacy
    :param D: delta values for peer-to-peer privacy
    :param R: Radius of the Euclidean ball in which the data vectors lie
    """

    n = P.shape[0]
    A = torch.distributions.Uniform(0, 0.1).sample((n, n))
    A = torch.where(P > 0, A, 0)

    T = torch.zeros_like(P)
    for i in range(n):
        for j in range(n):
            if j != i:
                T[i][j] = (
                    torch.sqrt(2 * torch.log(1.25 / D[i][j]))
                    * 2
                    * A[i][j]
                    * R
                    / E[i][j]
                )

    reg_noise = 1e-2
    sigma = torch.max(T).float() + reg_noise

    return A, sigma


class ZeroClipper(object):
    """Clip the weights to zero if they are negative"""

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):

        if hasattr(module, "A"):
            w = module.A.data
            nn.ReLU(inplace=True)(w)


def evaluate_log_barriers(
    A: torch.Tensor,
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.float,
    R: torch.float,
):
    """Compute the log-barrier penalty values for a given set of weights and privacy constraints
    :param A: Matrix of probabilities for intermittent connectivity amongst clients
    :param E: epsilon values for peer-to-peer privacy
    :param D: delta values for peer-to-peer privacy
    :param sigma: Privacy noise variance
    :param R: Radius of the Euclidean ball in which the data vectors lie
    """

    n = A.shape[0]  # Number of clients
    Bt = torch.zeros_like(A)  # log-barrier penalties
    Ct = torch.zeros_like(A)  # inequality-constraint slack

    for i in range(n):
        for j in range(n):
            if j != i:
                Ct[i][j] = (
                    E[i][j] * sigma
                    - torch.sqrt(2 * torch.log(1.25 / D[i][j])) * 2 * A[i][j] * R
                )
                Bt[i][j] = -torch.log(Ct[i][j])

    return Bt, Ct


def plot_losses(losses, title, xlabel, ylabel, outfile):
    """Plots loss from training process"""

    plt.plot(losses, color="blue", markersize=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
