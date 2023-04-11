# Misc utility modules

import torch
from torch import nn
import matplotlib.pyplot as plt
from loguru import logger


def evaluate_tiv(
    p: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
):
    """Evaluate the upper bound on topology induced variance
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param A: Matrix of weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param R: Radius of the Euclidean ball in which the data vectors lie
    """

    n = len(p)

    # Validate inputs
    assert n == A.shape[0] == A.shape[1], "p and A dimension mismatch!"
    assert n == P.shape[0] == P.shape[1], "p and P dimension mismatch!"

    # TIV
    S = 0

    # First term (squared terms)
    for i in range(n):
        for l in range(n):
            for j in range(n):
                S += p[j] * (1 - p[j]) * P[i][j] * P[l][j] * A[i][j] * A[l][j]

    # Second term (cross-terms)
    for i in range(n):
        for j in range(n):
            S += P[i][j] * p[j] * (1 - P[i][j]) * A[i][j] * A[i][j]

    # Third term due to correlation between reciprocal links
    for i in range(n):
        for l in range(n):
            assert P[i][l] == P[l][i], "Matrix P must be symmetric."
            E = P[i][l]
            S += p[i] * p[l] * (E - P[i][l] * P[l][i]) * A[l][i] * A[i][l]

    # Compute the bias terms
    s = torch.zeros(n)
    for i in range(n):
        for j in range(n):
            s[i] += p[j] * P[i][j] * A[i][j]

    # Fourth term due to bias
    S += torch.sum(s - 1) ** 2

    return S * (R**2) / n**2


def evaluate_piv(p: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor, d: int):
    """
    Evaluates the privacy induced variance
        :param d: Dimension of the data vectors
    """

    n = P.shape[0]  # number of clients

    t = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                t = t + p[j] * P[i][j]

    return (sigma**2 * d) / (n**2) * t


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
                T[i][j] = torch.sqrt(2 * torch.log(1.25 / D[i][j])) * (
                    2 * A[i][j] * R / E[i][j]
                )

    reg_noise = 1e-2
    sigma = torch.max(T) + reg_noise

    return A, sigma


def init_weights_from_priv_noise(
    P: torch.Tensor,
    E: torch.Tensor,
    D: torch.Tensor,
    R: torch.float,
    sigma: torch.float,
):
    """Initialize random weights that respect the network topology and feasible privacy noise
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param E: epsilon values for peer-to-peer privacy
    :param D: delta values for peer-to-peer privacy
    :param R: Radius of the Euclidean ball in which the data vectors lie
    :param sigma: Initial value of privacy noise
    """

    n = P.shape[0]
    A = torch.zeros_like(P)

    for i in range(n):
        for j in range(n):
            if P[i][j] > 0:
                ulim = ((E[i][j] * sigma) / (2 * R)) / torch.sqrt(
                    2 * torch.log(1.25 / D[i][j])
                )
                A[i][j] = torch.distributions.Uniform(0, ulim.item()).sample()

    return A


class ZeroClipper(object):
    """Clip the weights to zero if they are negative"""

    def __init__(self, proj, frequency=1):
        self.frequency = frequency
        self.proj = proj  # variable to project

    def __call__(self, module):
        if hasattr(module, "node_weights") and self.proj == "node_weights":
            w = module.node_weights.data
            nn.ReLU(inplace=True)(w)

        elif hasattr(module, "sigma_param") and self.proj == "sigma":
            s = module.sigma_param.data
            nn.ReLU(inplace=True)(s)


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
                Ct[i][j] = E[i][j] * sigma - torch.sqrt(
                    2 * torch.log(1.25 / D[i][j])
                ) * (2 * A[i][j] * R)
                Bt[i][j] = -torch.log(Ct[i][j])

    return Bt, Ct


def check_privacy_constraints(
    A: torch.Tensor,
    sigma: torch.Tensor,
    E: torch.Tensor,
    D: torch.Tensor,
    R: torch.float,
):
    """Check if a given pair (A, sigma) is feasible for a given set of privacy constraints"""

    n = A.shape[0]

    T_max = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                t = torch.sqrt(2 * torch.log(1.25 / D[i][j])) * (
                    2 * A[i][j] * R / E[i][j]
                )
                if T_max < t:
                    T_max = t

    if sigma < T_max:
        return (False, T_max)
    else:
        return (True, T_max)


def plot_losses(losses, title, xlabel, ylabel, outfile):
    """Plots loss from training process"""

    plt.plot(losses, color="blue", markersize=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
