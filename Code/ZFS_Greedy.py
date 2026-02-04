"""
ZFS_Greedy.py

This script applies a classical greedy algorithm to compute
zero forcing sets (ZFS) for a collection of large graphs.

For each graph:
  - The greedy ZFS algorithm is executed
  - Runtime is measured
  - The resulting ZFS and statistics are saved back into the .mat file

This script is primarily used for benchmarking and comparison
against learning-based ZFS methods.
"""

import os
import time
import numpy as np
import scipy.io as sio
from scipy.io import savemat

# Import greedy zero forcing algorithm
from greedy import *


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

greed_zfs = []     # Placeholder for greedy ZFS results (unused, kept for compatibility)
greed_time = 0     # Stores runtime of greedy algorithm

# Directory containing test graph files
base_dir = './data/'
data_path = os.path.join(base_dir, 'Large_Graph')

# List all graph files
val_mat_names = sorted(os.listdir(data_path))


# -------------------------------------------------------------------
# Run greedy ZFS on each graph
# -------------------------------------------------------------------

for idx in range(len(val_mat_names)):

    print(val_mat_names[idx])

    # Load graph data
    mat_contents = sio.loadmat(os.path.join(data_path, val_mat_names[idx]))

    # Extract adjacency matrix and convert to dense NumPy array
    adj_sparse = mat_contents['adj']
    adjacency = np.array(adj_sparse.todense())

    # Run greedy ZFS algorithm and measure runtime
    start_time = time.time()
    Z_1, Z_2 = Greedy_ZFS(adjacency)
    greed_time = time.time() - start_time

    print("Greedy ZFS size:", len(Z_2))

    # Number of nodes in greedy ZFS
    soln_greedy = len(Z_2)

    # -------------------------------------------------------------------
    # Save greedy results back into the .mat file
    # -------------------------------------------------------------------

    mat_contents['Greedy Time'] = greed_time
    mat_contents['Greedy Z1'] = Z_1
    mat_contents['Greedy Z2'] = Z_2
    mat_contents['soln_greedy'] = soln_greedy

    savemat(os.path.join(data_path, val_mat_names[idx]), mat_contents)
