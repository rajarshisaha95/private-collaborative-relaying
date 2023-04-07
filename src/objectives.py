# Objective functions for optimization

import torch


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
    """Evaluate the log-barrier penalty corresponding to a particular node
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


def local_tiv_priv(
    p: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    R: torch.float,
    eta: torch.float,
    E: torch.Tensor,
    D: torch.Tensor,
    sigma: torch.Tensor,
    node_idx: int,
    node_weights: torch.Tensor,
):
    """Evaluate the contribution of a node's weights to the TIV with the log-barrier penalty
    :param eta: Regularization strength of log barrier penalty
    """

    assert node_weights.requires_grad == True, "Node weights are not trainable!"

    return eta * local_tiv(
        p=p, P=P, A=A, R=R, node_idx=node_idx, node_weights=node_weights
    ) + local_log_barrier(
        E=E, D=D, sigma=sigma, node_idx=node_idx, node_weights=node_weights, R=R
    )
