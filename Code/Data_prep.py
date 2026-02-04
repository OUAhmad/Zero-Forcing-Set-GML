import numpy as np
import os
import shutil

# Large Graphs
src_dir_path = './rgraph_mat_files/'
dst_dir_path =  './data/data_gcn/large_graphs/'
dst_dir_path_1 =  './data/data_gcn/hybrid/'

src = os.listdir(src_dir_path)

np.random.seed(5252)
np.random.shuffle(src)


for i in range(int(0.8*len(src))):
    shutil.copyfile(src_dir_path + src[i], dst_dir_path + 'train/' + src[i])
    shutil.copyfile(src_dir_path + src[i], dst_dir_path_1 + 'train/' + src[i])
print(i)
for i in range(int(0.8*len(src)), len(src)):
    shutil.copyfile(src_dir_path + src[i], dst_dir_path + 'test/' + src[i])
    shutil.copyfile(src_dir_path + src[i], dst_dir_path_1 + 'test/' + src[i])
print(i)


# Small Graphs
src_dir_path = './data/Small_Graphs/'
dst_dir_path =  './data/data_gcn/small_graphs/'
dst_dir_path_1 =  './data/data_gcn/hybrid/'

src = os.listdir(src_dir_path)

np.random.seed(5252)
np.random.shuffle(src)


for i in range(int(0.8*len(src))):
    shutil.copyfile(src_dir_path + src[i], dst_dir_path + 'train/' + src[i])
    shutil.copyfile(src_dir_path + src[i], dst_dir_path_1 + 'train/' + src[i])
print(i)
for i in range(int(0.8*len(src)), len(src)):
    shutil.copyfile(src_dir_path + src[i], dst_dir_path + 'test/' + src[i])
    shutil.copyfile(src_dir_path + src[i], dst_dir_path_1 + 'test/' + src[i])
print(i)

