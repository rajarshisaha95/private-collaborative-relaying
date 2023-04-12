# Objective functions for optimization

import torch


def local_tiv(
    p: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    node_idx: int,
    node_weights: torch.Tensor,
):
    """Evaluate the contribution of a node's weights to the TIV -- a simplified objective function for Gauss-Seidel iterations
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param A: Matrix of collaboration weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param R: Radius of the Euclidean ball in which the data vectors lie
    :param node_idx: Node index for which local TIV is being evaluated
    :param node_weights: (Trainable) weights of node_idx node
    """

    assert node_weights.requires_grad == True, "Node weights are not trainable!"

    n = A.shape[0]  # Number of clients
    i = node_idx  # node-weight represents the weights allocated by node i for its neighbors

    # First term
    t1 = 0
    for j in range(n):
        t1 += p[j] * P[i][j] * (1 - p[j] * P[i][j]) * node_weights[j] ** 2
    t1 = t1 * (R**2)

    # Second term
    t2 = 0
    for l in range(n):
        if l != i:
            for j in range(n):
                t2 += p[j] * (1 - p[j]) * P[i][j] * P[l][j] * node_weights[j] * A[l][j]
    t2 = t2 * 2 * (R**2)

    # Third term
    t3 = 0
    for j in range(n):
        t3 += p[i] * p[j] * (P[i][j] - P[i][j] * P[j][i]) * (node_weights[j] ** 2)
    t3 = t3 * (R**2)

    # Compute biases
    s_i = 0  # Bias of i-th node
    for j in range(n):
        s_i += p[j] * P[i][j] * node_weights[j]

    s = torch.zeros(n)  # Bias of all nodes
    for l in range(n):
        if l != i:
            for j in range(n):
                s[l] += p[j] * P[l][j] * A[l][j]

    t4 = (s_i - 1) ** 2 + 2 * (s_i - 1) * (torch.sum(s) - n + 1)

    return t1 + t2 + t3 + t4


def local_log_barrier(
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.Tensor,
    node_idx: int,
    node_weights: torch.Tensor,
    R: torch.float,
):
    """Evaluate the log-barrier penalty for peer-to-peer privacy constraint corresponding to a particular node
    :param E: epsilon values for peer-to-peer privacy
    :param D: delta values for peer-to-peer privacy
    :param sigma: Privacy noise variance
    :param node_idx: Node index for which local TIV is being evaluated
    :param node_weights: (Trainable) weights of node_idx node
    :param R: Radius of the Euclidean ball in which the data vectors lie
    """

    assert node_weights.requires_grad == True, "Node weights are not trainable!"

    n = E.shape[0]  # Number of clients
    i = node_idx

    t = 0
    for j in range(n):
        if j != i:
            t = t - torch.log(
                E[i][j] * sigma
                - torch.sqrt(2 * torch.log(1.25 / D[i][j])) * 2 * node_weights[j] * R
            )

    return t


def local_nonneg_weights_penalty(node_weights: torch.Tensor):
    """Evaluate the log-barrier penalty for the non-negative weight constraint"""

    assert node_weights.requires_grad == True, "Node weights are not trainable!"

    n = len(node_weights)  # number of nodes

    t = 0
    for j in range(n):
        t = t - torch.log(node_weights[j])
    return t


def local_tiv_priv(
    p: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    eta_pr: torch.float,
    eta_nnw: torch.float,
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.Tensor,
    node_idx: int,
    node_weights: torch.Tensor,
):
    """Evaluate the contribution of a node's weights to the TIV with the log-barrier penalty
    :param eta_pr: Regularization strength of log-barrier penalty for privacy constraint
    :param eta_nnw: Regularization strength of log-barrier penalty for non-negative weights constraint
    """

    assert node_weights.requires_grad == True, "Node weights are not trainable!"

    return (
        local_tiv(p=p, P=P, A=A, R=R, node_idx=node_idx, node_weights=node_weights)
        + (1 / eta_pr)
        * local_log_barrier(
            E=E, D=D, sigma=sigma, node_idx=node_idx, node_weights=node_weights, R=R
        )
        + (1 / eta_nnw) * local_nonneg_weights_penalty(node_weights=node_weights)
    )


def piv(p: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor, d: int):
    """
    Evaluates the privacy induced variance
        :param d: Dimension of the data vectors
    """

    assert sigma.requires_grad == True, "Privacy noise is not trainable!"

    n = P.shape[0]  # number of clients

    t = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                t = t + p[j] * P[i][j]

    return (sigma**2 * d) / (n**2) * t


def log_barrier_penalty(
    A: torch.Tensor,
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.float,
    R: torch.float,
):
    """
    Evaluates the cumulative log-barrier penalty across all peer-to-peer privacy constraints
    """

    assert sigma.requires_grad == True, "Privacy noise is not trainable!"

    n = A.shape[0]
    B = 0  # cumulative log-barrier penalty

    for i in range(n):
        for j in range(n):
            if j != i:
                B = B - torch.log(
                    E[i][j] * sigma
                    - torch.sqrt(2 * torch.log(1.25 / D[i][j])) * 2 * A[i][j] * R
                )

    return B


def nonneg_priv_noise_penalty(sigma: torch.Tensor):
    """Evaluate the log-barrier penalty for the non-negative privacy noise constraint"""

    assert sigma.requires_grad == True, "Privacy noise is not trainable!"

    return -torch.log(sigma)


def piv_log_barrier(
    p: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    eta_pr: torch.float,
    eta_nnp: torch.float,
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.Tensor,
    d: int,
):
    """
    Evaluates the log-barrier penalized privacy induced variance
    :param eta_nnp: Regularization strength of log-barrier penalty for non-negative privacy noise constraint
    """

    return (
        piv(p=p, P=P, sigma=sigma, d=d)
        + (1 / eta_pr) * log_barrier_penalty(A=A, E=E, D=D, sigma=sigma, R=R)
        + (1 / eta_nnp) * nonneg_priv_noise_penalty(sigma=sigma)
    )
