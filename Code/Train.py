"""
train.py

This script trains a deep Graph Convolutional Network (GCN) model for
computing zero forcing sets (ZFS) in graphs. The model is trained to
predict node-level membership in a zero forcing set using graph
structure alone.

The implementation follows the framework proposed in:

"A Graph Machine Learning Framework to Compute Zero Forcing Sets in Graphs"
IEEE Transactions on Network Science and Engineering

Training is performed on small graph instances and evaluated using
distance-aware loss functions.
"""

from __future__ import division
from __future__ import print_function

import sys
import os
import time
import pickle
from copy import deepcopy

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tqdm import tqdm

# -------------------------------------------------------------------
# Module imports
# -------------------------------------------------------------------

# Add local modules directory to path
sys.path.append('%s/modules' % os.path.dirname(os.path.realpath(__file__)))

from modules.utils import *
from models import GCN_DEEP_DIVER
import modules.size_est
import size_est

# -------------------------------------------------------------------
# Global hyperparameters
# -------------------------------------------------------------------

N_bd = 32                 # Feature dimensionality (featureless nodes)
ITERATIONS = 4000         # Training iterations per epoch
RANDOM_SPLIT = 0.90       # Train/validation split ratio

# -------------------------------------------------------------------
# Dataset paths
# -------------------------------------------------------------------

# Directory containing training graph .mat files
data_path = './data/data_gcn/small_graphs/train/'
train_mat_names = os.listdir(data_path)

NUM_OF_GRAPHS = len(train_mat_names)

# Directory to save trained models and logs
model_dir = './gcn_models/small_graph_DCE_inv/'

# -------------------------------------------------------------------
# TensorFlow flags (hyperparameters)
# -------------------------------------------------------------------

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn_cheby', 'Model type.')
flags.DEFINE_string('loss', 'Dist_CE_loss', 'Loss function.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_integer('epochs', 60, 'Number of training epochs.')
flags.DEFINE_integer('hidden1', 32, 'Hidden layer size.')
flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate.')
flags.DEFINE_float('weight_decay', 5e-4, 'L2 regularization weight.')
flags.DEFINE_integer('early_stopping', 50, 'Early stopping patience.')
flags.DEFINE_integer('max_degree', 1, 'Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'Number of GCN layers.')

# -------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------

num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER

# Placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)),
    'labels': tf.placeholder(tf.float32, shape=(None, 2)),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'distance': tf.placeholder(tf.float32, shape=(None, None))
}

# Initialize model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# -------------------------------------------------------------------
# GPU configuration
# -------------------------------------------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# -------------------------------------------------------------------
# Evaluation function
# -------------------------------------------------------------------

def evaluate(features, support, labels, placeholders, distance):
    """
    Evaluates the model on validation data.

    Returns
    -------
    loss : float
    accuracy : float
    runtime : float
    softmax_output : ndarray
    """
    t_test = time.time()
    feed_dict = construct_feed_dict_up(
        features, support, labels, distance, placeholders
    )
    outs = sess.run(
        [model.loss, model.accuracy, model.outputs_softmax],
        feed_dict=feed_dict
    )
    return outs[0], outs[1], time.time() - t_test, outs[2]


# -------------------------------------------------------------------
# Session initialization and checkpoint loading
# -------------------------------------------------------------------

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(model_dir)
if ckpt:
    print('Loaded checkpoint:', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# -------------------------------------------------------------------
# Training statistics storage
# -------------------------------------------------------------------

tr_all_loss = np.zeros(FLAGS.epochs)
val_all_loss = np.zeros(FLAGS.epochs)
tr_all_acc = np.zeros(FLAGS.epochs)
val_all_acc = np.zeros(FLAGS.epochs)
tim = np.zeros(FLAGS.epochs)

# -------------------------------------------------------------------
# Dataset split
# -------------------------------------------------------------------

np.random.seed(1000)
split = np.arange(NUM_OF_GRAPHS)
np.random.shuffle(split)

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------

epoch_times = []

for epoch in range(FLAGS.epochs):
    tr_loss, tr_acc, val_loss, val_acc = [], [], [], []

    if os.path.isdir(model_dir + "%04d" % epoch):
        continue

    os.makedirs(model_dir + "%04d" % epoch)
    epoch_times.append(time.time())

    for _ in tqdm(range(ITERATIONS)):
        graph_id = np.random.randint(NUM_OF_GRAPHS)

        mat_contents = sio.loadmat(
            os.path.join(data_path, train_mat_names[graph_id])
        )

        adj = mat_contents['adj']

        # Compute or load distance matrix
        if 'distance' not in mat_contents:
            distance = nx.floyd_warshall_numpy(
                nx.from_numpy_matrix(adj.todense())
            )
            mat_contents['distance'] = distance
            sio.savemat(
                os.path.join(data_path, train_mat_names[graph_id]),
                mat_contents
            )
        else:
            distance = mat_contents['distance']

        # Load ZFS labels
        if 'Z2' in mat_contents:
            y = mat_contents['Z2'].astype(int).reshape((1, -1)).T
            yy = np.zeros((adj.shape[0], 1))
            yy[y] = 1
        else:
            yy = mat_contents['sol'].T

        # Randomly sample one ZFS realization
        y_sample = yy[:, np.random.randint(yy.shape[1])]
        y_train = np.column_stack((1 - y_sample, y_sample))

        # Featureless node features
        features = sp.lil_matrix(np.ones((adj.shape[0], N_bd)))
        features = preprocess_features(features)

        support = simple_polynomials(adj, FLAGS.max_degree)

        # Validation or training
        if graph_id in split[int(RANDOM_SPLIT * NUM_OF_GRAPHS):]:
            loss, acc, _, _ = evaluate(
                features, support, y_train, placeholders, distance
            )
            val_loss.append(loss)
            val_acc.append(acc)
        else:
            feed_dict = construct_feed_dict_up(
                features, support, y_train, distance, placeholders
            )
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            _, loss, acc, _ = sess.run(
                [model.opt_op, model.loss, model.accuracy, model.outputs],
                feed_dict=feed_dict
            )
            tr_loss.append(loss)
            tr_acc.append(acc)

    # Epoch summary
    tr_all_loss[epoch] = np.mean(tr_loss)
    val_all_loss[epoch] = np.mean(val_loss)
    tr_all_acc[epoch] = np.mean(tr_acc)
    val_all_acc[epoch] = np.mean(val_acc)
    tim[epoch] = time.time() - epoch_times[-1]

    print(
        f"{epoch+1:03d} | "
        f"train_loss={tr_all_loss[epoch]:.5f} "
        f"train_acc={tr_all_acc[epoch]:.5f} "
        f"val_loss={val_all_loss[epoch]:.5f} "
        f"val_acc={val_all_acc[epoch]:.5f} "
        f"time={tim[epoch]:.2f}s"
    )

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------

results = {
    'tr_all_loss': tr_all_loss,
    'val_all_loss': val_all_loss,
    'tr_all_acc': tr_all_acc,
    'val_all_acc': val_all_acc,
    'time': tim
}

with open(model_dir + "/results.pkl", "wb") as f:
    pickle.dump(results, f)

saver.save(sess, model_dir + "model.ckpt")
saver.save(sess, model_dir + "%04d/model.ckpt" % epoch)

print("Optimization Finished!")

# -------------------------------------------------------------------
# Plot training curves
# -------------------------------------------------------------------

plt.figure()
plt.plot(tr_all_loss, label="Training Loss")
plt.plot(val_all_loss, label="Validation Loss")
plt.legend()
plt.savefig(model_dir + "/loss.png")

plt.figure()
plt.plot(tr_all_acc, label="Training Accuracy")
plt.plot(val_all_acc, label="Validation Accuracy")
plt.legend()
plt.savefig(model_dir + "/acc.png")
