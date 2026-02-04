from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append('%s/modules' % os.path.dirname(os.path.realpath(__file__)))

import scipy
from copy import deepcopy
import scipy.io as sio
import time
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import random
from greedy import *
from os.path import exists
import pandas as pd
from scipy.io import savemat

import modules.size_est
import size_est
from modules.utils import *
from models import GCN_DEEP_DIVER

tf.disable_eager_execution()

N_bd = 500
ITERATIONS = 10000
RANDOM_SPLIT = 0.90
RANDOMNESS_FACTOR = 0.5

model_dir = './model'

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('loss','Dist_CE_Soln_loss', 'Loss Function')  # 'CE_loss', 'Dist_CE_loss', 'Dist_CE_Soln_loss'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')

# Some pre-processing
num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'distance': tf.placeholder(tf.float32, shape=(None, None)),
    'solution_nodes': tf.placeholder(tf.float32, shape=(None))
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Define model evaluation function
def evaluate(features, support, labels,  placeholders, distance, solution_nodes):
    t_test = time.time()
    feed_dict_val = construct_feed_dict_up(features, support, labels, distance, solution_nodes, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

# Init variables
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

# ckpt=tf.train.get_checkpoint_state("result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32")
ckpt=tf.train.get_checkpoint_state(model_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# Train GCN model
distance = np.zeros((1, 1))

gnn_time_list, greedy_time_list = [], []
predicted_zfs, greedy_zfs = [], []


    
# Load data
mat_contents = sio.loadmat('sample.mat')
try:
    adj = mat_contents['adj'].todense()
except:
    adj = mat_contents['adj']
nn = adj.shape[0]

# If the maximum degree in Graph is more than 500, the model can not handle that
if np.max(np.sum(adj, axis = 1)) > 499:
    print("Maximum Degree exceeds model limit")
    sys.exit()

# Find the Greedy Solution
start_time = time.time()
greedy_Z1,greedy_Z2 = Greedy_ZFS(adj)
greedy_time = np.round(time.time() - start_time, 4)
y = np.array(greedy_Z2).astype(int).reshape((1, -1)).transpose()
yy = np.zeros((adj.shape[0], 1))
yy[y] = 1 

greedy_soln = np.sum(yy)

# Model parameter Initializations
nn, nr = yy.shape # number of nodes & results
yyr = yy[:,np.random.randint(0,nr)]
yyr_num = np.sum(yyr)

# Random Selection
zfs_size = int(size_est.predict_zfs_size(adj, model_file = './reg_model//Regressor.joblib'))

start_time = time.time()
partial_sol = np.array(random.sample(set(np.arange(nn)), k=int(zfs_size * RANDOMNESS_FACTOR)))
derived_set = ZF_Span(adj,partial_sol)
span = len(derived_set)

y = partial_sol.astype(int).reshape((1, -1)).transpose()
yy = np.zeros((adj.shape[0], 1))
yy[y] = 1 
yyr = yy

# Iterative Solution Completion using GCN
while (span != nn):
    solution_nodes = yyr
    ytrain = np.zeros(adj.shape[0])
    ytrain[0] = 1
    y_train = np.concatenate([1 - np.expand_dims(ytrain, axis=1), np.expand_dims(ytrain, axis=1)], axis=1)
    features = np.zeros([nn, N_bd])
    features = sp.lil_matrix(features)
    features = preprocess_features_updated(features, yyr, adj)
    support = simple_polynomials(adj, FLAGS.max_degree)

    loss, accuracy, time_taken, output = evaluate(features, support, y_train,  placeholders, distance, solution_nodes)
    out = np.array(output)[:, 1]
    N_pred = np.argmax(out)
    if N_pred in partial_sol:
        count = 2
        while (np.argpartition(out, -count)[-count] in partial_sol):
            count += 1
        N_pred = np.argpartition(out, -count)[-count]
    partial_sol = np.append(np.array(np.nonzero(yyr)[0]).reshape((-1, 1)), N_pred).tolist()
    derived_set = ZF_Span(adj,partial_sol)
    span = len(derived_set)

    y = np.array(partial_sol).astype(int).reshape((1, -1)).transpose()
    yy = np.zeros((adj.shape[0], 1))
    yy[y] = 1 
    yyr = yy

    
gnn_time = np.round(time.time() - start_time, 4)

# Redundancy Check
start_time = time.time()
Z_1 = copy.deepcopy(partial_sol);
Z = copy.deepcopy(partial_sol);

for i in range(len(Z_1)):
    v = Z_1[i];
    Z_temp = np.setdiff1d(Z,Z_1[i]);
    DZ_temp = ZF_Span(adj,Z_temp);
    if (len(DZ_temp) == nn):
        Z = Z_temp;

redun_time = np.round(time.time() - start_time, 4)
try:
    Z = Z.tolist()
except:
    Z = np.array(Z).reshape(-1).tolist()
    

# Print results
print("\n-----------------------------------------------------------------------------------------")
print("\n gnn time :" + "{:.5f}".format(gnn_time + redun_time))
print( "\n greedy time: " + "{:.5f}".format(greedy_time))
print("\n Greedy Solution Size: " + "{:.3f}".format(len(greedy_Z2)))
print("\n GNN Solution Size: " + "{:.3f}".format(len(Z)))
print("\n-----------------------------------------------------------------------------------------")


