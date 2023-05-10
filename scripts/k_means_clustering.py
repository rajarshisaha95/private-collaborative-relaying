import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets

from multiprocessing.dummy import Pool

from loguru import logger
import matplotlib.pyplot as plt
import argparse

from mean_estimation import (
    intermittent_sole_good_client_naive,
    intermittent_sole_good_client_pricer_full_colab,
)


def load_mnist_data(dataset_path="../datasets/mnist"):
    """Load the MNIST dataset
    :param dataset_path: Location to download the dataset
    Return: Downloaded dataset as a numpy array
    """

    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    X = mnist_train.data.numpy().reshape(-1, 28 * 28)

    return X


def load_cifar10_data(dataset_path="../datasets/cifar10"):
    """Load the CIFAR-10 dataset
    :param dataset_path: Location to download the dataset
    :return: Downloaded dataset as a tensor
    """
    transform = torchvision.transforms.ToTensor()
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    train_data = torch.cat([sample[0].flatten().unsqueeze(0) for sample in cifar10_train], dim=0)

    return train_data.numpy()


def split_dataset(X, n_machines, partition: str = "uniform"):
    """Split the dataset into n_machines subsets
    :param X: Complete dataset
    :param n_machines: No. of machine across which X is to be distributed
    :param partition: Dataset split type
    Return: List of (array like) datasets
    """

    if partition == "uniform":
        np.random.shuffle(X)  # Shuffle the rows of X randomly

    else:
        pass

    if n_machines > 1:
        logger.info(f"Splitting the dataset according to partition type: {partition}")

    # Sort and part (by default)
    n_samples = len(X)
    subset_size = n_samples // n_machines

    subsets = []
    for i in range(n_machines):
        start = i * subset_size
        end = (i + 1) * subset_size
        subsets.append(X[start:end])

    return subsets


def kmeans_plus_plus(X, k, seed=42):
    """Initialize the centroids using k-means++ algorithm.
    :param X: Data matrix of shape (num_samples, num_features)
    :param k: Number of clusters in k-means clustering
    :param seed: Random seed to do a different initial clustering
    Return centroids of the k clusters
    """

    # Initialize the first centroid uniformly at random
    centroids = [X[np.random.choice(len(X))]]

    # Choose the remaining centroids using the K-means++ initialization algorithm
    for i in range(1, k):
        # Compute distance of each point from the centroids
        distances = np.zeros((len(X), len(centroids)))
        for i in range(len(X)):
            for j in range(len(centroids)):
                distances[i][j] = np.linalg.norm(X[i] - centroids[j])

        # Compute the minimum distance of each datapoint and all the previously chosen centroids
        min_distances = np.min(distances, axis=1)

        # Add a small constant to the min_distances array to avoid dividing by zero
        min_distances += np.finfo(float).eps

        # Compute the probability of selecting each data point as the next centroid
        probabilities = min_distances / np.sum(min_distances)

        # Choose the next centroid
        index = np.random.choice(len(X), p=probabilities)
        centroids.append(X[index])

    return np.array(centroids)


def run_local_kmeans(data, centroids, k, max_iterations):
    """Run k-means locally on each machine
    :param data: Local data of the machine
    :param centroids: Initial centroids broadcasted from the parameter server
    :param k: Number of clusters
    :param max_iterations: Number of local k-means iteration
    Return: List of updated centroids
    """

    num_datapoints = data.shape[0]

    distances = np.zeros([num_datapoints, k])

    for i in range(max_iterations):
        logger.info(f"Local k-means iteration: {i}/{max_iterations}")

        # Compute the distance of each datapoint to each of the centroids
        for j in range(num_datapoints):
            for l in range(k):
                distances[j][l] = np.linalg.norm(data[j, :] - centroids[l, :])

        labels = np.argmin(distances, axis=1)

        for l in range(k):
            centroids[l, :] = np.mean(data[labels == l], axis=0)

    return centroids


def calculate_inertia(data, centroids):
    """Inertia measures the sum of squared distances of samples to their closest cluster center
    A lower value of inertia indicates better clustering performance.
    """
    # Number of clusters
    k = len(centroids)

    # Calculate distances between data points and centroids
    distances = np.zeros((len(data), k))
    for i in range(k):
        distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)

    # Assign each data point to the nearest centroid
    labels = np.argmin(distances, axis=1)

    # Calculate inertia
    inertia = 0
    for i in range(k):
        inertia += np.sum(distances[labels == i, i])

    return inertia


def distributed_kmeans(
    X: np.ndarray = None,
    n_machines: int = 1,
    k: int = 1,
    max_iterations: int = 100,
    partition: str = "uniform",
    consensus_type: str = "perfect",
):
    """Performs distributed kmeans clustering on each of the machines by splitting the dataset
    :param X: Complete dataset
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_iterations: Number of iterations for k-means clustering
    :param partition: Dataset split type
    :param consensus_type: Topology parameters and algorithm used for computing global centroid
    Return inertia
    """

    # Split the dataset across different machines
    dim = X.shape[1]
    X_splits = split_dataset(X, n_machines, partition=partition)

    # Initialize the local centroids on each machine using k-means++
    logger.info(f"Initializing local centroids")
    local_centroids = [kmeans_plus_plus(X_i, k) for X_i in X_splits]

    # Run local k-means cluster on each machine
    updated_local_centroids = [None for _ in range(n_machines)]
    for i in range(n_machines):
        logger.info(f"Running local k-means on machine {i}/{n_machines}")
        updated_local_centroids[i] = run_local_kmeans(
            data=X_splits[i],
            centroids=local_centroids[i],
            k=k,
            max_iterations=max_iterations,
        )

    # Compute the average of the new centroids to get the global centroid
    if consensus_type == "perfect":
        global_centroid = np.mean(np.array(updated_local_centroids), axis=0)

    elif consensus_type == "intermittent_sole_good_client_naive":
        # Intermittent connectivity of clients to PS.
        #  No collaboration amongst clients
        logger.info(
            f"Computing global centroid naively with intermittent connectivity."
        )

        global_centroid = np.zeros([k, dim])

        for c_idx in range(k):
            T = np.zeros([n_machines, dim])

            for mc_idx in range(n_machines):
                T[mc_idx, :] = updated_local_centroids[mc_idx][c_idx, :]

            global_centroid[c_idx, :] = intermittent_sole_good_client_naive(
                data=T, n_machines=n_machines
            )

        logger.info(f"Estimated global centroid: {global_centroid}")
        logger.info(
            f"True global centroid: {np.mean(np.array(updated_local_centroids), axis=0)}"
        )
        logger.info(
            f"Estimation error (across all centroids): {np.linalg.norm(global_centroid - np.mean(np.array(updated_local_centroids), axis=0), ord='fro')}"
        )

    elif consensus_type == "intermittent_sole_good_client_pricer_full_colab":
        # Intermittent connectivity of clients to PS and with each other
        # Full collaboration amongst clients -- no privacy concerns
        logger.info(
            f"Computing global centroid collaboratively with intermittent connectivity: Full collaboration with no privacy constraints."
        )

        global_centroid = intermittent_sole_good_client_pricer_full_colab(
            updated_local_centroids
        )

        logger.info(f"Estimated global centroid: {global_centroid}")
        logger.info(
            f"True global centroid: {np.mean(np.array(updated_local_centroids), axis=0)}"
        )
        logger.info(
            f"Estimation error (across all centroids): {np.linalg.norm(global_centroid - np.mean(np.array(updated_local_centroids), axis=0), ord='fro')}"
        )

    else:
        logger.info(f"Consensus type not implemented! Defaulting to perfect consensus")
        global_centroid = np.mean(np.array(updated_local_centroids), axis=0)

    # Evaluate clustering accuracy
    inertia = calculate_inertia(X, global_centroid)

    return inertia


def compute_relative_inertia(
    X: np.ndarray = None,
    n_machines: int = 1,
    k: int = 1,
    max_iterations: int = 10,
    partition: str = "uniform",
    consensus_type: str = "perfect",
):
    """Compute inertia relative to clustering on a single machine
    :param X: Complete dataset
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_iterations: Number of iterations for k-means clustering
    :param partition: Dataset split type
    :param consensus_type: Topology parameters and algorithm used for computing global centroid
    Return relative inertia
    """

    logger.info(f"Doing distributed kmeans clustering...")
    inertia_distr = distributed_kmeans(
        X=X,
        n_machines=n_machines,
        k=k,
        max_iterations=max_iterations,
        partition=partition,
        consensus_type=consensus_type,
    )

    logger.info(f"Doing centralized kmeans clustering...")
    inertia_central = distributed_kmeans(
        X=X, n_machines=1, k=k, max_iterations=max_iterations
    )

    return inertia_distr / inertia_central


def simulate_kmeans(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_iterations: int = 100,
    num_realizations: int = 1,
    partition: str = "uniform",
    consensus_type: str = "perfect",
):
    """Simulate kmeans for different realizations and return average relative inertia
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_iterations: Number of iterations for k-means clustering
    :param num_realizations: Number of realizations to average inertia over
    :param consensus_type: Topology parameters and algorithm used for computing global centroid
    Return average relative inertia
    """

    rel_inertia = []

    if dataset == "mnist":
        logger.info(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        logger.info(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()
    
    else:
        logger.info(f"Dataset not configured!")

    for r in range(num_realizations):
        logger.info(f"Realization index: {r}")
        np.random.seed((r + 1) * 1000)
        relative_inertia = compute_relative_inertia(
            X=X,
            n_machines=n_machines,
            k=k,
            max_iterations=max_iterations,
            partition=partition,
            consensus_type=consensus_type,
        )
        rel_inertia.append(relative_inertia)
        logger.info(f"Realization {r}, Relative inertia: {relative_inertia}")

    return rel_inertia


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_machines", type=int, default=10, help="Number of machines")
    parser.add_argument("--k", type=int, default=10, help="Value of k")
    parser.add_argument(
        "--max_iterations", type=int, default=10, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--num_realizations", type=int, default=5, help="Number of realizations"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset being clustered"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="uniform",
        help="Type of dataset split across clients",
    )
    parser.add_argument(
        "--consensus_type",
        type=str,
        default="perfect",
        help="Topology parameters and algorithm used for computing global centroid",
    )

    args = parser.parse_args()

    n_machines = args.n_machines
    k = args.k
    max_iterations = args.max_iterations
    num_realizations = args.num_realizations
    dataset = args.dataset
    partition = args.partition
    consensus_type = args.consensus_type

    logger.info(
        f"K-means clustering: Dataset: {dataset}, Partition: {partition}, Consensus_type: {consensus_type}"
    )
    logger.info(
        f"n_machines = {n_machines}, k = {k}, max_iterations = {max_iterations}"
    )

    inertia_arr = simulate_kmeans(
        dataset=dataset,
        n_machines=n_machines,
        k=k,
        max_iterations=max_iterations,
        num_realizations=num_realizations,
        partition=partition,
        consensus_type=consensus_type,
    )

    logger.info(
        f"K-means clustering: Dataset: {dataset}, Partition: {partition}, Consensus_type: {consensus_type}"
    )
    logger.info(
        f"n_machines = {n_machines}, k = {k}, max_iterations = {max_iterations}"
    )
    logger.info(
        f"Inertia: Mean = {np.mean(inertia_arr):.3f}, Std. dev = {np.std(inertia_arr):.3f}"
    )
