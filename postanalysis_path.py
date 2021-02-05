import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import networkx as nx

parser = argparse.ArgumentParser(
    'Find shortest paths along domains in residues.')
parser.add_argument('--num-residues', type=int, default=77,
                    help='Number of residues of the PDB.')
parser.add_argument('--windowsize', type=int, default=56,
                    help='window size')
parser.add_argument('--dist-threshold', type=int, default=12,
                    help='threshold for shortest distance')
parser.add_argument('--filename', type=str, default='logs/out_probs_train.npy',
                    help='File name of the probs file.')
parser.add_argument('--source-node', type=int, default=46,
                    help='source residue of the PDB')
parser.add_argument('--outputfilename', type=str, default='logs/source46.txt',
                    help='File of shortest path from source to targets')
args = parser.parse_args()


# According to the distribution of learned edges between residues, we calculated the shortest path
# from mutation site to the residues in the active loop.

def getEdgeResults(threshold=False):
    a = np.load(args.filename)
    b = a[:, :, 1]
    c = a[:, :, 2]
    d = a[:, :, 3]

    # There are four types of edges, eliminate the first type as the non-edge
    probs = b+c+d
    # For default residue number 77, residueR2 = 77*(77-1)=5852
    residueR2 = args.num_residues*(args.num_residues-1)
    probs = np.reshape(probs, (args.windowsize, residueR2))

    # Calculate the occurence of edges
    edges_train = probs/args.windowsize

    results = np.zeros((residueR2))
    for i in range(args.windowsize):
        results = results+edges_train[i, :]

    if threshold:
        # threshold, default 0.6
        index = results < (args.threshold)
        results[index] = 0

    # Calculate prob for figures
    edges_results = np.zeros((args.num_residues, args.num_residues))
    count = 0
    for i in range(args.num_residues):
        for j in range(args.num_residues):
            if not i == j:
                edges_results[i, j] = results[count]
                count += 1
            else:
                edges_results[i, j] = 0

    return edges_results


def dist_cal(AA1, AA2):
    """
    calculate CA-CA distance
    """
    dist = np.sqrt(np.square(
        AA1['x'] - AA2['x']) + np.square(AA1['y'] - AA2['y']) + np.square(AA1['z'] - AA2['z']))
    return dist


# Load distribution of learned edges
edges_results = getEdgeResults()

dists_matrix = list()
for i in range(1, 2):
    tmp_matrix = np.zeros((args.num_residues, args.num_residues))
    df = pd.read_csv('data/pdb/ca_%d.pdb' % i, sep='\s+', names=[
        'ATOM', 'n1', 'CA', 'AA', 'Chain', 'n2', 'x', 'y', 'z', 'n3', 'n4', 'C'])  # temparory DataFrame
    tdf = df.iloc[1:(args.num_residues+1)]
    for ind_1 in range(args.num_residues-1):
        for ind_2 in range(ind_1+1, args.num_residues):
            AA1 = tdf.iloc[ind_1]
            AA2 = tdf.iloc[ind_2]
            tmp_dist = dist_cal(AA1, AA2)
            tmp_matrix[ind_1, ind_2] = tmp_dist
            tmp_matrix[ind_2, ind_1] = tmp_dist
    dists_matrix.append(tmp_matrix)
dists_matrix = np.array(dists_matrix)
dists_mean = np.zeros((args.num_residues, args.num_residues))
for ind_1 in range(args.num_residues-1):
    for ind_2 in range(ind_1+1, args.num_residues):
        dists_mean[ind_1, ind_2] = dists_matrix[:, ind_1, ind_2].mean()
        dists_mean[ind_2, ind_1] = dists_matrix[:, ind_2, ind_1].mean()
np.save('logs/dists_mean_12.npy', dists_mean)
# if the distance is larer than 12, then ignore
filtered_edges = np.where(dists_mean > args.dist_threshold, 1, edges_results)
np.save('logs/filtered_edges_12.npy', filtered_edges)

edges = np.load('logs/filtered_edges_12.npy')
# The network is directed
edges_list = list()
# Default: i->j
for i in range(args.num_residues):
    for j in range(args.num_residues):
        if i != j:
            edges_list.append((i, j, {'weight': filtered_edges[j, i]}))
MDG = nx.MultiDiGraph()
MDG.add_edges_from(edges_list)

source_node = args.source_node  # set source node
target_nodes = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]  # set target nodes
out_file = args.outputfilename

path_dict = dict()
for tn in target_nodes:
    path_dict[tn] = []
    path_nodes = nx.dijkstra_path(MDG, source_node, tn)
    path_length_list = []  # save the length of shorest path
    path_length_list.append(nx.dijkstra_path_length(MDG, source_node, tn))
    path_dict[tn].append('shortest_path : ' + '->'.join(list(map(str, path_nodes))) +
                         ' : ' + str(nx.dijkstra_path_length(MDG, source_node, tn)))
    if len(path_nodes) > 2:
        for ipn in range(1, len(path_nodes)):
            tmp_MDG = MDG.copy()
            tmp_MDG.remove_edge(path_nodes[ipn-1], path_nodes[ipn])
            tmp_path_nodes = nx.dijkstra_path(tmp_MDG, source_node, tn)
            path_length_list.append(
                nx.dijkstra_path_length(tmp_MDG, source_node, tn))
            path_dict[tn].append('remove(%d->%d) : ' % (path_nodes[ipn-1], path_nodes[ipn]) + '->'.join(
                list(map(str, tmp_path_nodes))) + ' : ' + str(nx.dijkstra_path_length(tmp_MDG, source_node, tn)))
        # According to the shortest path length from small to large
        path_dict[tn] = np.array(path_dict[tn])[np.argsort(path_length_list)]
with open(out_file, 'w') as f:
    for k, v in path_dict.items():
        f.write('target node : %d\n' % k)
        for tmp_path in v:
            f.write('\t\t' + tmp_path + '\n')
