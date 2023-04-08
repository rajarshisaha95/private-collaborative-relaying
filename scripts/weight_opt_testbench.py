# Setups to test the weight optimization procedure under various constraints

import fire
import pathlib
import torch
import matplotlib.pyplot as plt

from loguru import logger

from src.utils import (
    init_random_weights,
    init_random_weights_priv_noise,
    plot_losses,
    evaluate_log_barriers,
)
from src.optimization import NodeWeightsUpdate, NodeWeightsUpdatePriv
from src.objectives import evaluate_tiv


def simple_network_test():

    # Network parameters
    num_clients = 5
    p = torch.Tensor([0.5, 0.4, 0.3, 0.8, 0.9])  # Connectivity to PS
    P = torch.zeros(num_clients, num_clients)  # Connectivity between clients
    P.fill_diagonal_(1)
    for i in range(num_clients):
        P[i][(i + 1) % num_clients] = 0.9
        P[i][(i - 1) % num_clients] = 0.9

    logger.info(f"P is:\n {P}")

    # Data parameters
    radius = 1.0  # Domain: Euclidean ball

    # Optimizer parameters
    num_iters_gs = 100  # Gauss-Seidel iteration
    weights_lr = 0.005
    weights_num_iters = 100

    # Initialization values
    weights_init = init_random_weights(P=P)

    weights_orig = weights_init.clone().detach()

    weights_opt, losses = NodeWeightsUpdate().gauss_seidel_weight_opt(
        num_iters=num_iters_gs,
        A_init=weights_init,
        p=p,
        P=P,
        R=radius,
        weights_lr=weights_lr,
        weights_num_iters=weights_num_iters,
    )

    logger.info(f"Weights before optimization:\n{weights_orig}")

    logger.info(f"Weights after optimization:\n{weights_opt}")

    logger.info(
        f"ColRel Gauss-Seidel final loss value: {evaluate_tiv(p=p, P=P, A=weights_opt, R=radius)}"
    )

    # Save the plot
    with torch.no_grad():
        outfile = pathlib.Path("misc") / f"weight_opt_gauss-seidel_TIV.png"
        plot_losses(
            losses=losses,
            xlabel="Gauss-Seidel iterations",
            ylabel="Topology Induced Variance",
            title="Weight optimization using Gauss-Seidel",
            outfile=outfile,
        )


def simple_network_test_private():

    # Network parameters
    num_clients = 5
    p = torch.Tensor([0.5, 0.4, 0.3, 0.8, 0.9])  # Connectivity to PS
    pc = 0.9
    P = pc * torch.ones(num_clients, num_clients)  # Connectivity between clients
    P.fill_diagonal_(1)

    logger.info(f"P is:\n {P}")

    # Privacy parameters
    delta = 1e-3
    D = delta * P
    eps1 = 1e3  # Trustworthy neighbors
    eps2 = 1e-2  # Non-trustworthy neighbors
    E = eps2 * torch.ones([num_clients, num_clients])
    for i in range(num_clients):
        E[i][i] = eps1
        E[i][(i - 1) % num_clients] = eps1
        E[i][(i + 1) % num_clients] = eps1

    logger.info(f"E is:\n {E}")

    # Data parameters
    radius = 1.0  # Domain: Euclidean ball

    # Regularization parameters
    eta = 100

    # Optimizer parameters
    num_iters_gs = 20  # Gauss-Seidel iteration
    weights_lr = 0.001
    weights_num_iters = 1000

    # Initialization values
    weights_init, sigma_init = init_random_weights_priv_noise(P=P, E=E, D=D, R=radius)

    weights_orig = weights_init.clone().detach()

    weights_opt, losses = NodeWeightsUpdatePriv().gauss_seidel_weight_opt_priv(
        num_iters=num_iters_gs,
        A_init=weights_init,
        p=p,
        P=P,
        R=radius,
        E=E,
        D=D,
        eta=eta,
        sigma=sigma_init,
        weights_lr=weights_lr,
        weights_num_iters=weights_num_iters,
    )

    logger.info(f"Weights before optimization:\n{weights_orig}")
    logger.info(f"Weights after optimization:\n{weights_opt}")

    B_init, C_init = evaluate_log_barriers(
        A=weights_orig, E=E, D=D, sigma=sigma_init, R=radius
    )
    logger.info(f"Log-barriers before optimization:\n{B_init}")
    logger.info(f"Constraint values before optimization:\n{C_init}")

    B_opt, C_opt = evaluate_log_barriers(
        A=weights_opt, E=E, D=D, sigma=sigma_init, R=radius
    )
    logger.info(f"Log-barriers after optimization:\n{B_opt}")
    logger.info(f"Constraint values after optimization:\n{C_opt}")

    logger.info(
        f"ColRel Gauss-Seidel final loss value: {evaluate_tiv(p=p, P=P, A=weights_opt, R=radius)}"
    )

    # Save the plot
    with torch.no_grad():
        outfile = pathlib.Path("misc") / f"weight_opt_gauss-seidel_TIV_priv.png"
        plot_losses(
            losses=losses,
            xlabel="Gauss-Seidel iterations",
            ylabel="Topology Induced Variance",
            title="Weight optimization with Privacy constraints using Gauss-Seidel",
            outfile=outfile,
        )


if __name__ == "__main__":
    # fire.Fire(simple_network_test)
    fire.Fire(simple_network_test_private)