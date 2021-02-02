import time
import numpy as np
import argparse
from copy import deepcopy
from scipy import interpolate

parser = argparse.ArgumentParser('Generate features from pdb')
parser.add_argument('--MDfolder', type=str, default="data/pdb/",
                    help='folder of pdb MD')
parser.add_argument('--residue-size', type=int, default=77,
                    help='residue size of the MD pdb')
parser.add_argument('--feature-size', type=int, default=6,
                    help='The number of features used in study( position (X,Y,Z) + velocity (X,Y,Z) ).')
args = parser.parse_args()


def read_feature_file(filename, feature_size=1, gene_size=10, timestep_size=21):
    """
    Read single expriments of all time points
    """
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


def read_feature_MD_file(filename, timestep_size, feature_size, residue_size, interval):
    """
    Read single expriments of all time points
    """
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
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[5])-1] = float(words[6])
                feature[numStep, 1, int(words[5])-1] = float(words[7])
                feature[numStep, 2, int(words[5])-1] = float(words[8])
            elif(line.startswith("ATOM") and words[2] == "CA" and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(
                    words[5])-1] = float(words[6])-feature[numStep, 0, int(words[5])-1]
                feature[numStep, 4, int(
                    words[5])-1] = float(words[7])-feature[numStep, 1, int(words[5])-1]
                feature[numStep, 5, int(
                    words[5])-1] = float(words[8])-feature[numStep, 2, int(words[5])-1]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    return feature


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
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[4]) >= aa_start and int(words[4]) <= aa_end and flag):
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[4])-aa_start] = float(words[5])
                feature[numStep, 1, int(words[4])-aa_start] = float(words[6])
                feature[numStep, 2, int(words[4])-aa_start] = float(words[7])
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[4]) >= aa_start and int(words[4]) <= aa_end and nflag):
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
    # print(feature.shape)
    return feature


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
                numStep = int(modelNum/interval)
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


def convert_dataset_md_single(MDfolder, startIndex, experiment_size, timestep_size, feature_size, residue_size, interval, window_start, window_end, aa_start, aa_end):
    """
    Convert in single md file in single skeleton
    """
    features = list()
    edges = list()

    for i in range(startIndex, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        for j in range(window_start, window_end+1):
            # print(str(i)+" "+str(j))
            features.append(read_feature_MD_file_slidingwindow(MDfolder+"ca_"+str(
                i)+".pdb", timestep_size, feature_size, residue_size, interval, j, aa_start, aa_end))
            edges.append(np.zeros((residue_size, residue_size)))
    print("***")
    print(len(features))
    print("###")
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


MDfolder = args.MDfolder
feature_size = args.feature_size
residue_size = args.residue_size


# Generate training/validating/testing
print("Generate Train")
features, edges = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                            feature_size=feature_size, residue_size=residue_size, interval=60, window_start=1, window_end=56, aa_start=1, aa_end=77)

np.save('data/features.npy', features)
np.save('data/edges.npy', edges)


print("Generate Valid")
features_valid, edges_valid = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                                        feature_size=feature_size, residue_size=residue_size, interval=60, window_start=1, window_end=56, aa_start=1, aa_end=77)

np.save('data/features_valid.npy', features_valid)
np.save('data/edges_valid.npy', edges_valid)


print("Generate Test")
features_test, edges_test = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                                      feature_size=feature_size, residue_size=residue_size, interval=100, window_start=1, window_end=56, aa_start=1, aa_end=77)
np.save('data/features_test.npy', features_test)
np.save('data/edges_test.npy', edges_test)
