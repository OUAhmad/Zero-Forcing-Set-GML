"""
Train_Regressor.py

This script trains a regression model to estimate the size of the
optimal zero forcing set (ZFS) of a graph using simple structural
features extracted from its adjacency matrix.

The trained model is used as part of the graph machine learning
framework described in:

"A Graph Machine Learning Framework to Compute Zero Forcing Sets in Graphs"
IEEE Transactions on Network Science and Engineering
"""

import csv
import os
import math
import time

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


# Path containing MATLAB (.mat) graph files
data_path = "./mat_files/"


def feature_extraction(Adjacency):
    """
    Extracts graph-level structural features from an adjacency matrix.

    Features include:
        - Number of nodes |V|
        - Number of edges |E|
        - Minimum degree
        - Five smallest node degrees
        - Five largest node degrees

    Parameters
    ----------
    Adjacency : numpy.ndarray or scipy sparse matrix
        Adjacency matrix of the graph.

    Returns
    -------
    feat : list
        List of extracted graph features.
    """
    # Convert adjacency matrix to NetworkX graph
    try:
        G = nx.convert_matrix.from_scipy_sparse_matrix(Adjacency)
    except Exception:
        G = nx.from_numpy_matrix(Adjacency)

    feat = []

    # Basic graph statistics
    feat.append(G.number_of_nodes())
    feat.append(G.number_of_edges())

    # Degree-based features
    degrees = sorted([val for (_, val) in G.degree()])
    feat.append(degrees[0])  # Minimum degree

    # Add 5 smallest and 5 largest degrees
    for i in range(1, 5):
        feat.append(degrees[i])
        feat.append(degrees[-i])

    return feat


def prepare_data(data_path, feat_file="./reg_model/index.txt"):
    """
    Reads all graph .mat files, extracts features, and saves them
    along with ground-truth ZFS sizes into a CSV file.

    Parameters
    ----------
    data_path : str
        Directory containing .mat graph files.
    feat_file : str
        Output CSV file storing features and labels.
    """
    val_mat_names = sorted(os.listdir(data_path))

    with open(feat_file, "w") as writeFile:
        csvwriter = csv.writer(writeFile)

        for idx, file_name in enumerate(val_mat_names):
            print(idx, file_name)

            # Load MATLAB graph file
            mat_contents = sio.loadmat(os.path.join(data_path, file_name))

            # Extract graph features
            data = feature_extraction(mat_contents["adj"])

            # Retrieve ground-truth ZFS size
            try:
                z2 = mat_contents["Z2_size"][0]
            except Exception:
                # Fallback for different dataset formats
                if "Z2" in mat_contents:
                    z2 = mat_contents["Z2_size"]
                else:
                    z2 = np.sum(mat_contents["sol"])

            data.append(z2)
            csvwriter.writerow(data)


def training(feat_file="./reg_model/index.txt",
             model_file="./reg_model/Regressor.joblib"):
    """
    Trains a regression model to estimate ZFS size from graph features
    and saves the trained model to disk.

    Parameters
    ----------
    feat_file : str
        CSV file containing graph features and labels.
    model_file : str
        Path to save the trained regression model.
    """
    # Load feature dataset
    data = pd.read_csv(feat_file)
    print("Shape of data:", data.shape)

    # Separate features and target variable
    X = data.iloc[:, 0:11]
    y = data.iloc[:, 11]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=789
    )

    # Train Random Forest regressor
    model = RandomForestRegressor(n_estimators=500)
    print("Training regression model...")
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Save trained model
    dump(model, model_file)

    # Report error metrics
    print("MAE train:", mean_absolute_error(y_train, train_preds))
    print("MAE test:", mean_absolute_error(y_test, test_preds))
    print("RMSE train:", math.sqrt(mean_squared_error(y_train, train_preds)))
    print("RMSE test:", math.sqrt(mean_squared_error(y_test, test_preds)))

    # Plot predictions vs ground truth
    fig, ax = plt.subplots()
    ax.scatter(range(len(test_preds)), test_preds, marker="*", label="Predicted")
    ax.scatter(range(len(test_preds)), y_test, label="Ground Truth")
    ax.set(
        xlabel="Test Sample Index",
        ylabel="ZFS Size",
        title="ZFS Size Prediction (Test Set)",
    )
    ax.legend()
    ax.grid()

    fig.savefig("./reg_model/train.png")


def testing(mat_file, data_path,
            model_file="./reg_model/Regressor.joblib"):
    """
    Predicts the ZFS size for a single graph using the trained model.

    Parameters
    ----------
    mat_file : str
        Name of the .mat graph file.
    data_path : str
        Directory containing the graph file.
    model_file : str
        Path to the trained regression model.

    Returns
    -------
    float
        Predicted ZFS size.
    """
    mat_contents = sio.loadmat(os.path.join(data_path, mat_file))
    data = feature_extraction(mat_contents["adj"])

    clf = load(model_file)
    prediction = clf.predict(np.array(data).reshape(1, -1))

    return prediction[0]


# -------------------- Main Execution --------------------

# Generate feature dataset
prepare_data(data_path)

# Train and save regression model
training("./reg_model/index.txt")

# Example testing loop (commented)
# val_mat_names = sorted(os.listdir(data_path))
# for name in val_mat_names:
#     print(testing(name, data_path))
