from scipy import interpolate
from copy import deepcopy
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser('Generate features from pdb')
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=500000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--prior-strength', type=float, default=0.1,
                    help='prior strength')

args = parser.parse_args()


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all


def read_feature_file(filename, feature_size=1, gene_size=10, timestep_size=21):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, gene_size))

    time_count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if(time_count >= 0 and time_count < timestep_size):
                words = line.split()
                data_count = 0
                for word in words:
                    feature[time_count, 0, data_count] = word
                    data_count += 1
            time_count += 1
    f.close()
    # Use interpole
    feature = timepoint_sim(feature, 4)
    return feature


def read_feature_Residue_file(filename):
    resdict = {}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split(",")
            if count > 0:
                feature = np.zeros((len(words)-1))
                for i in range(len(words)-1):
                    feature[i] = words[i+1]
                resdict[words[0]] = feature
            count = count+1
    return resdict

# timestep_size=50
# feature_size = 6 loc + vel
#residue_size = 20
#interval = 1000


def read_feature_MD_file(filename, timestep_size, feature_size, residue_size, interval):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, residue_size))

    flag = False
    nflag = False
    modelNum = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if(line.startswith("MODEL")):
                modelNum = int(words[1])
                if (modelNum % interval == 1):
                    flag = True
                if (modelNum % interval == 2):
                    nflag = True
            elif(line.startswith("ATOM") and words[2] == "CA" and flag):
                # print(line)
                numStep = int(modelNum/interval)
                # print(line)
                # # print(words[1]+"\t"+words[5])
                # print(modelNum)
                # print(str(modelNum)+"\t"+str(numStep))
                feature[numStep, 0, int(words[4])-1] = float(words[5])
                feature[numStep, 1, int(words[4])-1] = float(words[6])
                feature[numStep, 2, int(words[4])-1] = float(words[7])
            elif(line.startswith("ATOM") and words[2] == "CA" and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(
                    words[4])-1] = float(words[5])-feature[numStep, 0, int(words[4])-1]
                feature[numStep, 4, int(
                    words[4])-1] = float(words[6])-feature[numStep, 1, int(words[4])-1]
                feature[numStep, 5, int(
                    words[4])-1] = float(words[7])-feature[numStep, 2, int(words[4])-1]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    return feature

# timestep_size=50
# feature_size = 6 loc + vel
#residue_size = 20
#interval = 1000
#window_choose = 1


def read_feature_MD_file_slidingwindow(filename, timestep_size, feature_size, residue_size, interval, window_choose, aa_start, aa_end):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, residue_size))

    flag = False
    nflag = False
    modelNum = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if(line.startswith("MODEL")):
                modelNum = int(words[1])
                if (modelNum % interval == window_choose):
                    flag = True
                if (modelNum % interval == (window_choose+1)):
                    nflag = True
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[1]) >= aa_start and int(words[1]) <= aa_end and flag):
                # print(line)
                numStep = int(modelNum/interval)
                # print(line)
                # # print(words[1]+"\t"+words[5])
                # print(modelNum)
                # print(str(modelNum)+"\t"+str(numStep))
                # print(str(numStep)+"\t"+words[4]+"\t"+words[5])
                feature[numStep, 0, int(words[4])-aa_start] = float(words[5])
                feature[numStep, 1, int(words[4])-aa_start] = float(words[6])
                feature[numStep, 2, int(words[4])-aa_start] = float(words[7])
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[1]) >= aa_start and int(words[1]) <= aa_end and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(
                    words[4])-aa_start] = float(words[5])-feature[numStep, 0, int(words[4])-aa_start]
                feature[numStep, 4, int(
                    words[4])-aa_start] = float(words[6])-feature[numStep, 1, int(words[4])-aa_start]
                feature[numStep, 5, int(
                    words[4])-aa_start] = float(words[7])-feature[numStep, 2, int(words[4])-aa_start]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    print(feature.shape)
    return feature

# timestep_size=50
# feature_size = 6 loc + vel
#residue_size = 20
#interval = 1000


def read_feature_MD_file_resi(filename, resDict, feature_size, residue_size, timestep_size, interval):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, residue_size))

    flag = False
    nflag = False
    modelNum = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if(line.startswith("MODEL")):
                modelNum = int(words[1])
                if (modelNum % interval == 1):
                    flag = True
                if (modelNum % interval == 2):
                    nflag = True
            elif(line.startswith("ATOM") and words[2] == "CA" and flag):
                # print(line)
                numStep = int(modelNum/interval)
                # print(line)
                # # print(words[1]+"\t"+words[5])
                # print(modelNum)
                feature[numStep, 0, int(words[4])-1] = float(words[5])
                feature[numStep, 1, int(words[4])-1] = float(words[6])
                feature[numStep, 2, int(words[4])-1] = float(words[7])
                featureResi = resDict[words[3]]
                for i in range(6, 6+featureResi.shape[0]):
                    feature[numStep, i, int(words[4])-1] = featureResi[i-6]

            elif(line.startswith("ATOM") and words[2] == "CA" and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(
                    words[4])-1] = float(words[5])-feature[numStep, 0, int(words[4])-1]
                feature[numStep, 4, int(
                    words[4])-1] = float(words[6])-feature[numStep, 1, int(words[4])-1]
                feature[numStep, 5, int(
                    words[4])-1] = float(words[7])-feature[numStep, 2, int(words[4])-1]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    return feature


def read_edge_file(filename, gene_size):
    edges = np.zeros((gene_size, gene_size))
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            data_count = 0
            for word in words:
                edges[count, data_count] = word
                data_count += 1
            count += 1
    f.close()
    return edges


def convert_dataset(feature_filename, edge_filename, experiment_size=5):
    features = list()

    edges = np.zeros((experiment_size, experiment_size))

    for i in range(1, experiment_size+1):
        features.append(read_feature_file(feature_filename+"_"+str(i)+".txt"))

    count = 0
    with open(edge_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            data_count = 0
            for word in words:
                edges[count, data_count] = word
                data_count += 1
            count += 1
    f.close()

    features = np.stack(features, axis=0)
    edges = np.tile(edges, (features.shape[0], 1)).reshape(
        features.shape[0], features.shape[3], features.shape[3])
    return features, edges


def convert_dataset_sim(feature_filename, edge_filename, experiment_size=5, gene_size=5, sim_size=50000):
    features = list()

    edges = np.zeros((gene_size, gene_size))

    for i in range(1, experiment_size+1):
        features.append(read_feature_file(
            feature_filename+"_"+str(i)+".txt"), gene_size=5)

    count = 0
    with open(edge_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            data_count = 0
            for word in words:
                edges[count, data_count] = word
                data_count += 1
            count += 1
    f.close()

    features = np.stack(features, axis=0)

    features_out = np.zeros(
        (sim_size, features.shape[1], features.shape[2], features.shape[3]))
    edges_out = np.zeros((sim_size, gene_size, gene_size))

    for i in range(sim_size):
        index = np.random.permutation(np.arange(experiment_size))
        num = np.random.permutation(np.arange(experiment_size))[0]
        features_out[i, :, :, :] = features[num, :, :, :][:, :, index]
        edges_out[i, :, :] = edges[index, :][:, index]

    # Add noise
    features_out = features_out + \
        np.random.randn(
            sim_size, features.shape[1], features.shape[2], features.shape[3])
    return features_out, edges_out


def convert_dataset_md(feature_filename, startIndex, experiment_size, timestep_size, feature_size, residue_size, interval):
    features = list()
    edges = list()

    for i in range(startIndex, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        features.append(read_feature_MD_file(feature_filename+"smd"+str(i) +
                                             ".pdb", timestep_size, feature_size, residue_size, interval))
        edges.append(np.zeros((residue_size, residue_size)))

    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    return features, edges

#features: loc + vel
# use sliding window to add


def convert_dataset_md_single(MDfolder, MDfilename, timestep_size, feature_size, residue_size, interval, window_start, window_end, aa_start, aa_end):
    features = list()
    edges = list()

    for i in range(window_start, window_end+1):
        features.append(read_feature_MD_file_slidingwindow(MDfolder+MDfilename,
                                                           timestep_size, feature_size, residue_size, interval, i, aa_start, aa_end))
        edges.append(np.zeros((residue_size, residue_size)))
    print("***")
    print(len(features))
    print("###")
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    return features, edges

#features: loc + vel + more


def convert_dataset_md_more(feature_filename, AAfeature_filename, experiment_size=1, timestep_size=50, feature_size=10, residue_size=20, interval=1000):
    features = list()
    edges = list()

    resDict = read_feature_Residue_file(AAfeature_filename)

    for i in range(1, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        features.append(read_feature_MD_file_resi(feature_filename+"smd"+str(i)+".pdb",
                                                  resDict, timestep_size=50, feature_size=10, residue_size=20, interval=1000))
        edges.append(np.zeros((residue_size, residue_size)))

    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    return features, edges


def timepoint_sim(feature, fold):
    # hard code now,fold=4
    # feature_shape: [timestep, feature_size, gene]
    step = 1/fold
    timestep = feature.shape[0]
    genes = feature.shape[2]
    x = np.arange(timestep)
    xnew = np.arange(0, (timestep-1)+step, step)
    feature_out = np.zeros((xnew.shape[0], 1, genes))
    for gene in range(genes):
        y = feature[:, 0, gene]
        tck = interpolate.splrep(x, y, s=0)
        ynew = interpolate.splev(xnew, tck, der=0)
        feature_out[:, 0, gene] = ynew
    return feature_out


MDfolder = "./data/pdb/"
MDfilename = 'toy.pdb'

features, edges = convert_dataset_md_single(MDfolder, MDfilename, timestep_size=50,
                                            feature_size=6, residue_size=77, interval=100, window_start=0, window_end=1, aa_start=11, aa_end=93)

np.save('features.npy', features)
np.save('edges.npy', edges)

features_val, edges_val = convert_dataset_md_single(MDfolder, MDfilename, timestep_size=50,
                                                    feature_size=6, residue_size=77, interval=30, window_start=0, window_end=1, aa_start=11, aa_end=93)

np.save('features_val.npy', features_val)
np.save('edges_val.npy', edges_val)

print("Generate Test")
features_test, edges_test = convert_dataset_md_single(MDfolder, MDfilename, timestep_size=100,
                                                      feature_size=6, residue_size=77, interval=33, window_start=0, window_end=1, aa_start=11, aa_end=93)

np.save('features_test.npy', features_test)
np.save('edges_test.npy', edges_test)
