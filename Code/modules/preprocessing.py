import networkx as nx
import numpy as np
from scipy.io import savemat
import os
import time
import sys

sys.path.append( '%s/modules' % os.path.dirname(os.path.realpath(__file__)) )

from greedy import *

dir = "./"
index_file = dir + "merged_solution1.txt"
edge_files = dir + 'newData/'
mat_files = dir + 'mat_files/'

def compute_greedy_for_dataset(index_file, edge_files):\
    # Create a file with name index_file and store the Z1_size, Z2_size, time taken by greedy, and Z2 for each file in edge_files directory
    f = open(index_file, "a")

    path = edge_files
    folder = os.fsencode(path)
    
    for file in os.listdir(folder):
        
        filename = path+os.fsdecode(file)
        if (filename.find('90_0') == -1):
            continue
        G = nx.read_edgelist( filename )
        
        A = nx.adjacency_matrix(G).todense()
        start = time.time()
        z1, z2 = Greedy_ZFS(A)
        end = time.time()
        line = [ len(z1), len(z2), end-start ]
        line.extend(z2)
        # f.write(filename)
        f.write(os.path.basename(filename))
        f.write(',')
        f.write(str(line) )
        f.write('\n')
        print(line)
    f.close()

def get_graph(name, dir_path):
    # Takes in the directory path and the graph edgelist name and returns the graph and the Adjacency Matrix (Sparse Form)
    G = nx.read_edgelist(os.path.join(dir_path, name))
    A = nx.adjacency_matrix(G)
    return G, A

def get_graph_contents(data, edgelist_path):
    # Takes in the line of index file and returns the Graph contents
    a = data.split(',')
    if len(a) >= 3:
        fname = a[0]
        G, A = get_graph(fname, edgelist_path)
        Z1_size = a[1]
        Z2_size = a[2]
        greedy_time = a[3]
        Z2 = a[4:]
        return  fname, G, A, Z1_size, Z2_size, greedy_time, Z2
    else:
        return 

def create_mat_files(index_file, edgelist_path, mat_files = './mat_files/'):
    # Takes in the paths for index file, graphs edge list directory and the mat files directory and creates mat files for all graphs in mat_files directory
    file1 = open(index_file,"r+") 
    count = 0
    for i in file1.readlines():
        try:
            fname, G, A, Z1_size, Z2_size, greedy_time, Z2 = get_graph_contents(i, edgelist_path)
            mdic = {"adj": A, "Z1_size": Z1_size, "Z2_size": Z2_size, "greedy_time": greedy_time, "Z2": Z2}
            savemat(mat_files + fname + ".mat", mdic)
        except:
            print("******** Data Unavailable for :", fname, " ********")
        print("done for: ", fname)

try: 
    os.mkdir(mat_files) 
except OSError as error: 
    print(error)

if os.path.exists(index_file) == False:
    compute_greedy_for_dataset(index_file, edge_files)	

# compute_greedy_for_dataset(index_file, edge_files)
# create_mat_files(index_file, edge_files, mat_files)

