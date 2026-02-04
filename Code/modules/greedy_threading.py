import time
from threading import Thread
import numpy as np
import copy
import multiprocessing

CORES = multiprocessing.cpu_count() - 10

def call_script(args):
    ZF_Span_thread.call((args))

# DZ = span
# Z = zero_f_set
# z = size_of_span
# DZ = span


def Greedy_ZFS_Extend(A, Z_1):
# -------------------------------------------------------------------------
#  This function takes a graph and a partial solution as input and returns the output of a Greedy
#  ZFS and a modified greedy ZFS algorithm.
#  Input: A (adjacency matrix of an undirected graph), Z_1 a partial solution
#  Output: Z_1 (nodes included in ZFS as a result of greedy algorithm).
#        : Z_2 (If the ZFS returned by greedy (Z_1) is not minimal, then, we
#        simply remove redundant nodes from Z_1 iteratively to obtain a
#        maximal ZFS, which is Z_2. We can call it the output of modified
#        greedy algorithm.
# --------------------------------------------------------------------------
    n = A.shape[0];                                       # no. of nodes in a graph.
    zero_f_set = Z_1
    zero_f_set = zero_f_set[0].tolist();
    span = ZF_Span(A,zero_f_set);
    size_of_span = len(span);                    # initialization
    DZ_temp = np.zeros((n, n))
    f_values = np.zeros((n))
    t = []

    while (size_of_span < n):                                    # condition to check if we have obtained ZFS or not.
        # print("*************************************************************")
        # print(size_of_span)
        Potential_A = np.setdiff1d(range(n),span);      # possible choices of nodes that can be included in ZFS.        
        DZ_temp = np.zeros((n, n))
        f_values = np.zeros((n))

        for i in range(len(Potential_A)):
            v = Potential_A[i];
            DZ1 = span + [v]
            t.append(Thread(target=ZF_Span_thread, args=(A, DZ1, i, DZ_temp, f_values)))
            t[-1].start()
            time.sleep(0.0001)

            # temp_zfs = copy.deepcopy(span)
            # temp_zfs.append( v )

            # temp_span = ZF_Span(A, temp_zfs);                # Set of nodes that would be colored by including v in a ZFS.
            # f_values[i] = len(temp_span);      

        x, y = np.max(f_values), np.argmax(f_values)
        zero_f_set.append(Potential_A[y]);
        span = ZF_Span(A,zero_f_set);
        size_of_span = len(span);
    
    Z_1 = zero_f_set;
        
    for i in range(len(Z_1)):
        v = Z_1[i];
        Z_temp = np.setdiff1d(zero_f_set,Z_1[i]);
        DZ_temp = ZF_Span(A,Z_temp);
        if (len(DZ_temp) == n):
            zero_f_set = Z_temp;
        # print(len(DZ_temp))
    Z_2 = zero_f_set;
    return Z_1, Z_2

def Greedy_ZFS(A):
# -------------------------------------------------------------------------
#  This function takes a graph as input and returns the output of a Greedy
#  ZFS and a modified greedy ZFS algorithm.
#  Input: A (adjacency matrix of an undirected graph)
#  Output: Z_1 (nodes included in ZFS as a result of greedy algorithm).
#        : Z_2 (If the ZFS returned by greedy (Z_1) is not minimal, then, we
#        simply remove redundant nodes from Z_1 iteratively to obtain a
#        maximal ZFS, which is Z_2. We can call it the output of modified
#        greedy algorithm.
# --------------------------------------------------------------------------
    n = len(A);                                       # no. of nodes in a graph.
    Z = [];     DZ = [];    z = 0;                    # initialization
    DZ_temp = np.zeros((n, n))
    f_values = np.zeros((n))
    t = []
    while (z < n):                                    # condition to check if we have obtained ZFS or not.
        # print(z)
        Potential_A = np.setdiff1d(range(n),DZ);      # possible choices of nodes that can be included in ZFS.
        # f_values = np.zeros((len(Potential_A)));
        DZ_temp = np.zeros((n, n))
        f_values = np.zeros((n))
        for i in range(len(Potential_A)):
            v = Potential_A[i];
            DZ1 = DZ + [v]
            t.append(Thread(target=ZF_Span_thread, args=(A, DZ1, i, DZ_temp, f_values)))
            t[-1].start()
            # while len(t)>= CORES:
            #     if CORES > 100:
            #         for i in t[:10]:
            #             i.join()
            #     else:
            #         time.sleep(1)
            time.sleep(0.0001)
            # _thread.start_new_thread( ZF_Span, (A, DZ1, i))
            # DZ_temp = ZF_Span(A, DZ1, i);                # Set of nodes that would be colored by including v in a ZFS.
            # f_values[i] = len(DZ_temp);      
            # Wait for all threads to complete
        x, y = np.max(f_values), np.argmax(f_values)
        # print(f_values)
        # print(Potential_A)
        Z.append(Potential_A[y]);
        DZ = ZF_Span(A,Z);
        z = len(DZ);
    print("Hello")
    Z_1 = Z;

    for i in range(len(Z_1)):
        v = Z_1[i];
        Z_temp = np.setdiff1d(Z,Z_1[i]);
        DZ_temp = ZF_Span(A,Z_temp);
        if (len(DZ_temp) == n):
            Z = Z_temp;
    Z_2 = Z;
    return Z_1, Z_2

def ZF_Span_thread(A,S, j, DZ_temp, f_values):
# -------------------------------------------------------------------------
#  This function takes a set of initially black colored nodes, run the
#  coloring process, and returns the set of nodes that become colored as a
#  result of the coloring process.
#  Input: A (Adjacency matrix of the undirected graph)
#         S (Set of initially colored black nodes)
#  Output: Black (Set of nodes that become black as a result of the coloring
#  process).
# -------------------------------------------------------------------------
    n = len(A);                                               # no. of nodes in a graph.
    Black = list(S); White = np.setdiff1d(range(n),Black);    # initialization
    flag = True;  Black_tot = Black;                           # initialization
    while(flag):
        flag = False
        for i in range(len(Black)):
            b = Black[i];
            N_b = np.argwhere(A[b,:]);
            N_b_white = np.intersect1d(White,N_b);
            if (len(N_b_white)==1):
                Black_tot.append(N_b_white[0]);
                White = np.setdiff1d(range(n), Black_tot);
                flag = True;
    Black = Black_tot;
    DZ_temp[j, Black] = 1
    f_values[j] = len(Black)
    # return Black


def ZF_Span(A,S):
# -------------------------------------------------------------------------
#  This function takes a set of initially black colored nodes, run the
#  coloring process, and returns the set of nodes that become colored as a
#  result of the coloring process.
#  Input: A (Adjacency matrix of the undirected graph)
#         S (Set of initially colored black nodes)
#  Output: Black (Set of nodes that become black as a result of the coloring
#  process).
# -------------------------------------------------------------------------
    n = len(A);                                               # no. of nodes in a graph.
    Black = list(S); White = np.setdiff1d(range(n),Black);    # initialization
    flag = True;  Black_tot = Black;                           # initialization
    while(flag):
        flag = False
        for i in range(len(Black)):
            b = Black[i];
            N_b = np.argwhere(A[b,:]);
            N_b_white = np.intersect1d(White,N_b);
            if (len(N_b_white)==1):
                Black_tot.append(N_b_white[0]);
                White = np.setdiff1d(range(n), Black_tot);
                flag = True;
    Black = Black_tot;
    return Black
