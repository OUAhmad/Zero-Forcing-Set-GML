import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time 
import pandas as pd
import math
import scipy.io as sio


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


data_path = "./mat_files/"


def feature_extraction(Adjacency):
    # Extracts the features of Graph from the mat file. 
    try:
        G = nx.from_numpy_matrix(Adjacency)
    except:
        G = nx.convert_matrix.from_scipy_sparse_matrix(Adjacency)
    feat = []
    feat.append( G.number_of_nodes() )
    feat.append( G.number_of_edges() )
    degrees = sorted( [val for (node, val) in G.degree()] )
    feat.append(degrees[0])
    for i in range(1,5):
        feat.append(degrees[i])
        feat.append(degrees[-i])
    return feat

def prepare_data(data_path, feat_file = 'index.txt'):
    # Takes in the mat files path and a file name to store the featrues and Z2_size. 
    # Saves a file that contains certain features of all the graphs
    val_mat_names = sorted(os.listdir(data_path))
    print("Extracting Features")
    with open(feat_file, 'w') as writeFile:
        csvwriter = csv.writer(writeFile)

        for id in range(len(val_mat_names)):
            mat_contents = sio.loadmat(data_path + val_mat_names[id])
            data = feature_extraction(mat_contents['adj'])
            try:
                z2 = mat_contents['Z2_size'][0]
            except:
                z2 = mat_contents['Z2_size']
            data.append(z2)
            csvwriter.writerow(data)


def train(feat_file = 'index.txt', model_file = 'Regressor.joblib'):
    # Takes in a csv file of features of the graphs and trains and save a regressor model for Z2_size estimation

    data = pd.read_csv(feat_file)
    print("Shape of data as read from the file : ", data.shape)
    #data.columns = [i.lstrip() for i in data.columns]
    #data = data.loc[:, data.columns != 'id']
    #data = data.apply(pd.to_numeric, errors='coerce')
    #data = data[(MIN_STRESS < data['v_stress']) & (data['v_stress'] < MAX_STRESS)]
    data_x = data.iloc[:, 0:11]
    data_y = data.iloc[:, 11]
    X_train, X_test, train_labels, test_labels = train_test_split(data_x, data_y, test_size=0.15, random_state=789)
    #lm = LinearRegression()
    lm = RandomForestRegressor(n_estimators=500)
    print("training the model...")
    lm.fit(X_train, train_labels)
    train_predictions = lm.predict(X_train)
    test_predictions = lm.predict(X_test)

    model = 'models'
    try: 
        os.mkdir(model) 
    except OSError as error: 
        print(error)
    dump(lm, model + "/" + model_file) 


    print("MAE train:",mean_absolute_error(train_labels, train_predictions ))
    print("MAE test:",mean_absolute_error(test_labels, test_predictions))

    print("MSE train:",math.sqrt(mean_squared_error( train_labels, train_predictions)))
    print("MSE test:",math.sqrt(mean_squared_error(test_labels, test_predictions)))

    fig, ax = plt.subplots()
    ax.scatter(range(len(test_predictions)), test_predictions, marker = '*', label = 'Test Value')
    ax.scatter(range(len(test_predictions)), test_labels, label = 'Predicted Value')
    ax.set(xlabel='test Value', ylabel='Predicted Value)',
        title='Testing Curve')
    ax.legend()
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
    return lm

def predict_zfs_size(adj,  model_file = './models/Regressor.joblib'):
    # Takes in a mat_file, calculates all the features, and returns a prediction on the size of optimal ZFS
    data = feature_extraction(adj)
    # data = feature_extraction(mat_contents['adj'])
    clf = load(model_file)
    try:
        return clf.predict(np.array(data).reshape((1,-1)))[0]
    except:
        return clf.predict(np.array(data).reshape((1,-1)))

# prepare_data(data_path)
# train(data_path)

# val_mat_names = sorted(os.listdir(data_path))
# for idx, id in enumerate(range(len(val_mat_names))):
    # mat_contents = sio.loadmat(data_path +  mat_file)
    # print(predict(mat_contents))






