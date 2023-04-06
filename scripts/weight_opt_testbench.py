# Setups to test the weight optimization procedure under various constraints

import fire
import pathlib
import torch
import matplotlib.pyplot as plt

from loguru import logger

from src.utils import init_random_weights, plot_losses
from src.optimization import gauss_seidel_weight_opt
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

    print("Hello!")

    weights_opt, losses = gauss_seidel_weight_opt(
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


if __name__ == "__main__":
    fire.Fire(simple_network_test)
