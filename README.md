# Mean Estimation with Collaborative Relaying under Privacy Constraints

This repository contains the implementation of __PriCER__: **Pri**vate **C**ollaborative **E**stimation via **R**elaying, as proposed in [Privacy Preserving Semi-Decentralized Mean Estimation over Intermittently-Connected Networks](https://drive.google.com/file/d/1BMZ5DjambHAEV3_cYHub7OkR6CBtZe_X/view?usp=sharing) by **Rajarshi Saha**, **Mohamed Seif**, **Michal Yemini**, **Andrea J. Goldsmith**, and **H. Vincent Poor**.

We consider the problem of privately estimating the mean of vectors distributed across different nodes of an unreliable wireless network, where communications between nodes can fail intermittently. A semi-decentralized setup is adopted, wherein to mitigate the impact of intermittently connected links, nodes can collaborate with their neighbors to compute a local consensus, which they relay to a central server. In such a setting, the communications between any pair of nodes must ensure that the privacy of the nodes is rigorously maintained to prevent unauthorized information leakage. We study the tradeoff between collaborative relaying and privacy leakage due to the data sharing among nodes and, subsequently, propose __PriCER__, a differentially private collaborative algorithm for mean estimation to optimize this tradeoff. The privacy guarantees of __PriCER__ arise (i) *implicitly*, by exploiting the inherent stochasticity of the flaky network connections, and (ii) *explicitly*, by adding Gaussian perturbations to the estimates exchanged by the nodes. Local and central privacy guarantees are provided against eavesdroppers who can observe different signals, such as the communications amongst nodes during local consensus and (possibly multiple) transmissions from the relays to the central server.

## Problem Formulation and Algorithm

1. Connectivity is modeled using Bernoulli random variables.

2. Peer-to-Peer privacy constraints: Communication fom node $i$ to node $j$ should satisfy local differential privacy.

3. The communication channel, i.e., connectivity probabilities between nodes and between a node and the paramter server, are assumed to be known beforehand for stage $0$.

### __PriCER__ Stage 0: (Pre-Processing) Joint weight and noise variance optimization

__Input__: Connectivity probabilities and privacy constraints, Maximal iterations: $T$

__Output__: Optimized collaboration weight matrix and privacy noise variance matrix

__Initialize__: Uniform initialization of weights and variances.

__For__ $t = 1$ to $T$:

- Take gradient descent steps
- Project onto cone constraints.


### __PriCER__ Stage 1: Local aggregation

__Input__: Collaboration weights and privacy noise variances

__Output__: Locally aggregated mean at every node

__For__ $i = 1$ to $n$:

- Locally generate data
- Transmit scaled and Gaussian perturbed signal to neighbors
- Receive signals from transmitting neighbors if channel is instantaneously present. Null signal otherwise.
- Aggregate local contribution
- Locally aggregate all received signals
- Transmit locally aggregated signal to the parameter server over an intermittent link


### __PriCER__ Stage 2: Global aggregation

__Input__: Received signals from each node. Null signal if channel is instaneously absent.

__Output__: Estimate of the mean at the server

__For__ $i = 1$ to $n$:

- Receive signals from successfully transmitting nodes

Globally aggregate the received signals




## 🛠 Setup
Run the following command to install all dependencies in a conda environment named `ColRel`. 
```
conda env create -n ColRel -f ColRel_env.yml
```
After installation, activate the environment with
```
conda activate ColRel
```

Note: __PriCER__ requires fairly standard dependencies. You may also choose to install them manually. 


## Repository structure

Detailed descriptions are provided as corresponding docstrings in the code. 

### `src/`

- `src/mean_estimation.py`: Distributedly computing mean, either naively, or using __PriCER__

- `src/objectives.py`: Objectives with respect to which weights and variances are optimized.

- `src/optimization.py`: Functions to implement joint weight and noise variance optimization.

- `src/topology.py`: Functions to implement different connectivity network topologies. This is initially run to save the adjacency matrix of the topologies as .npy files.

- `src/utils.py`: Auxiliary helper functions

### `scripts/`

- `scripts/k_means_clustering.py`: Auxiliary functions for processing k-means clustering experiments and evaluations

- `scripts/k_means_testbench.py`: Functions to implement distributed k-means clustering over different network topologies.

- `scripts/testbench.py` and `scripts.weight_opt_testbench.py`: Additional functions to replicate some experiments from the paper.