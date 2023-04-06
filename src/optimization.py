# Definitions of optimization objects

import torch
from torch import nn

from loguru import logger

from src.objectives import local_tiv, evaluate_tiv
from src.utils import ZeroClipper


class Model_NodeWeights(nn.Module):
    """Pytorch model for gradient optimization of node weights"""

    def __init__(
        self,
        A: torch.Tensor,
        node_idx: int,
        p: torch.Tensor,
        P: torch.Tensor,
        R: torch.float,
    ):

        super().__init__()
        self.i = node_idx
        self.A = A
        self.p = p
        self.P = P
        self.R = R

        # Make a copy of the node_idx row trainable and initialize to previous weights
        self.node_weights = nn.Parameter(A[self.i, :])

    def forward(self):
        """Contribution to MSE from the weights of a particular node"""

        forward_loss = local_tiv(
            p=self.p,
            A=self.A,
            P=self.P,
            R=self.R,
            node_idx=self.i,
            node_weights=self.node_weights,
        )

        return forward_loss


def training_loop_node_weights(model: nn.Module, optimizer, num_iters: int):
    """Training loop for torch model."""

    losses = []

    assert hasattr(model, "node_weights"), "Trainable node weights not found!"
    for i in range(num_iters):

        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Projection for non-negative weight constraint
        clipper = ZeroClipper()
        if i % clipper.frequency == 0:
            model.apply(clipper)

        losses.append(loss)

        if i % 100 == 0:
            logger.info(f"Iteration: {i}/{num_iters}")

    return losses


def update_node_weights(
    A: torch.Tensor,
    node_idx: int,
    p: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    weights_lr=torch.float,
    weights_num_iters=int,
):
    """Update the node_idx row of A (for Gauss-Seidel iterations)
    :param A: Matrix of probabilities for intermittent connectivity amongst clients
    :param node_idx: Node index for which weights are updated
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param R: Radius of the Euclidean ball in which the data vectors lie
    :param d: Dimension of the data vectors
    :param weights_lr: Learning rate for node weights
    :param weights_num_iters: Number of iterations for learning node weights
    return: Updated weight matrix with the node_idx row updated
    """

    logger.info(f"Updating weights of node: {node_idx}")

    # Create a node weight optimization model
    m = Model_NodeWeights(A=A, node_idx=node_idx, p=p, P=P, R=R)

    # Instantiate optimizer
    opt = torch.optim.Adam(m.parameters(), lr=weights_lr)

    # Run the optimization loop
    _ = training_loop_node_weights(model=m, optimizer=opt, num_iters=weights_num_iters)

    # Update the weights of node: node_idx
    m.node_weights.requires_grad = False
    A[node_idx, :] = m.node_weights

    del m

    return A


def gauss_seidel_weight_opt(
    num_iters: int,
    A_init: torch.Tensor,
    p: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    weights_lr=torch.float,
    weights_num_iters=int,
):
    """Iteratively optimize the weights of each node and the noise variance.
        :param num_iters: Number of passes over the optimization variables
        :param A_init: Initial weights
        :param p: Array of transmission probabilities from each of the clients to the PS
        :param P: Matrix of probabilities for intermittent connectivity amongst clients
        :param R: Radius of the Euclidean ball in which the data vectors lie
        :param d: Dimension of the data vectors
        :param weights_lr: Learning rate for node weights
        :param weights_num_iters: Number of iterations for learning node weights
    returns the optimized weight matrix
    """

    A = A_init
    n = A.shape[0]  # Number of clients

    losses = []

    for iters in range(num_iters):

        logger.info(f"Gauss-Seidel iteration: {iters}/{num_iters}")

        # Optimize over weights of node i keeping weights of other nodes fixed
        for i in range(n):
            A = update_node_weights(
                A=A,
                node_idx=i,
                p=p,
                P=P,
                R=R,
                weights_lr=weights_lr,
                weights_num_iters=weights_num_iters,
            )

        losses.append(evaluate_tiv(p=p, A=A, P=P, R=R))

    return A, losses
