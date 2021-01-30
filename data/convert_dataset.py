import time
import numpy as np
import argparse
from copy import deepcopy
from scipy import interpolate

parser = argparse.ArgumentParser()
#parser.add_argument('--simulation', type=str, default='springs',
#                    help='What simulation to generate.')
#parser.add_argument('--num-train', type=int, default=50000,
#                    help='Number of training simulations to generate.')
#parser.add_argument('--num-valid', type=int, default=10000,
#                    help='Number of validation simulations to generate.')
#parser.add_argument('--num-test', type=int, default=10000,
#                    help='Number of test simulations to generate.')
#parser.add_argument('--n-balls', type=int, default=10,
#                    help='Number of balls in the simulation.')
#parser.add_argument('--seed', type=int, default=42,
#                    help='Random seed.')
parser.add_argument('--prior-strength', type=float, default=0.1,
                    help='prior strength')

args = parser.parse_args()

# if args.simulation == 'springs':
#     sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
#     suffix = '_springs'
# elif args.simulation == 'charged':
#     sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
#     suffix = '_charged'
# else:
#     raise ValueError('Simulation {} not implemented'.format(args.simulation))

# suffix += str(args.n_balls)
# np.random.seed(args.seed)

# print(suffix)


def read_feature_file(filename, feature_size=1, gene_size=10, timestep_size=21):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, gene_size))

    time_count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if(time_count >= 0 and time_count<timestep_size):
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
    resdict={}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split(",")
            if count>0:
                feature = np.zeros((len(words)-1))
                for i in range(len(words)-1):
                    feature[i] = words[i+1]
                resdict[words[0]] = feature
            count=count+1
    return resdict

#timestep_size=50
#feature_size = 6 loc + vel
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
                if (modelNum%interval == 1):
                    flag = True
                if (modelNum%interval == 2):
                    nflag = True
            elif(line.startswith("ATOM") and words[2]=="CA" and flag):
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[5])-1] = float(words[6])
                feature[numStep, 1, int(words[5])-1] = float(words[7])
                feature[numStep, 2, int(words[5])-1] = float(words[8])
            elif(line.startswith("ATOM") and words[2]== "CA" and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(words[5])-1] = float(words[6])-feature[numStep, 0, int(words[5])-1]
                feature[numStep, 4, int(words[5])-1] = float(words[7])-feature[numStep, 1, int(words[5])-1]
                feature[numStep, 5, int(words[5])-1] = float(words[8])-feature[numStep, 2, int(words[5])-1]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    return feature

#timestep_size=50
#feature_size = 6 loc + vel
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
                if (modelNum%interval == window_choose):
                    flag = True
                if (modelNum%interval == (window_choose+1)):
                    nflag = True
            elif(line.startswith("ATOM") and words[2]=="CA" and int(words[4])>=aa_start and int(words[4])<=aa_end and flag ):
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[4])-aa_start] = float(words[5])
                feature[numStep, 1, int(words[4])-aa_start] = float(words[6])
                feature[numStep, 2, int(words[4])-aa_start] = float(words[7])
            elif(line.startswith("ATOM") and words[2]== "CA" and int(words[4])>=aa_start and int(words[4])<=aa_end and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(words[4])-aa_start] = float(words[5])-feature[numStep, 0, int(words[4])-aa_start]
                feature[numStep, 4, int(words[4])-aa_start] = float(words[6])-feature[numStep, 1, int(words[4])-aa_start]
                feature[numStep, 5, int(words[4])-aa_start] = float(words[7])-feature[numStep, 2, int(words[4])-aa_start]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    print(feature.shape)
    return feature

#timestep_size=50
#feature_size = 6 loc + vel
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
                if (modelNum%interval == 1):
                    flag = True
                if (modelNum%interval == 2):
                    nflag = True
            elif(line.startswith("ATOM") and words[2]=="CA" and flag):
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[4])-1] = float(words[5])
                feature[numStep, 1, int(words[4])-1] = float(words[6])
                feature[numStep, 2, int(words[4])-1] = float(words[7])
                featureResi = resDict[words[3]]
                for i in range(6,6+featureResi.shape[0]):
                    feature[numStep, i, int(words[4])-1] = featureResi[i-6]
                
            elif(line.startswith("ATOM") and words[2]== "CA" and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(words[4])-1] = float(words[5])-feature[numStep, 0, int(words[4])-1]
                feature[numStep, 4, int(words[4])-1] = float(words[6])-feature[numStep, 1, int(words[4])-1]
                feature[numStep, 5, int(words[4])-1] = float(words[7])-feature[numStep, 2, int(words[4])-1]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    return feature

def read_edge_file(filename, gene_size):
    edges = np.zeros((gene_size,gene_size))
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            data_count = 0
            for word in words:
                edges[count,data_count]= word
                data_count += 1
            count += 1
    f.close()
    return edges


def convert_dataset(feature_filename, edge_filename, experiment_size=5):
    features = list()   

    edges = np.zeros((experiment_size,experiment_size))

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
                edges[count,data_count]= word
                data_count += 1
            count += 1
    f.close()
    
    features = np.stack(features, axis=0)
    edges = np.tile(edges,(features.shape[0],1)).reshape(features.shape[0],features.shape[3],features.shape[3])
    return features, edges

def convert_dataset_sim(feature_filename, edge_filename, experiment_size=5, gene_size=5, sim_size=50000):
    features = list()   

    edges = np.zeros((gene_size,gene_size))

    for i in range(1, experiment_size+1):
        features.append(read_feature_file(feature_filename+"_"+str(i)+".txt"),gene_size=5)
    
    count = 0
    with open(edge_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            data_count = 0
            for word in words:
                edges[count,data_count]= word
                data_count += 1
            count += 1
    f.close()
    
    features = np.stack(features, axis=0)

    features_out = np.zeros((sim_size,features.shape[1],features.shape[2],features.shape[3]))
    edges_out = np.zeros((sim_size,gene_size,gene_size))

    for i in range(sim_size):
        index = np.random.permutation(np.arange(experiment_size))
        num = np.random.permutation(np.arange(experiment_size))[0] 
        features_out[i,:,:,:] = features[num,:,:,:][:,:,index]
        edges_out[i,:,:] = edges[index,:][:,index]
    
    # Add noise
    features_out = features_out + np.random.randn(sim_size,features.shape[1],features.shape[2],features.shape[3])
    return features_out, edges_out

def convert_dataset_Dream4(feature_filename, edge_filename, experiment_size=5, replic_size=5, gene_size=10, sim_size=50000, prior_strength=args.prior_strength):
    features = list()
    edges = list() 
    priors = list()

    for i in range(1, experiment_size+1):
        edge = read_edge_file(edge_filename+str(i)+".txt", gene_size)
        
        # Add prior
        prior_rand = np.random.rand(*edge.shape)
        prior = deepcopy(edge)
        index=np.where(prior_rand < prior_strength)

        # Change the groundtruth as the prior
        for j in np.arange(index[0].size):
	        prior[index[0][j],index[1][j]]=not edge[index[0][j],index[1][j]]        
        
        for j in range(1, replic_size+1):
            features.append(read_feature_file(feature_filename+str(i)+"/insilico_size10_"+str(i)+"_timeseries_experiment"+str(j)+".tsv",gene_size=10,timestep_size=21))
            edges.append(edge)
            priors.append(prior)             
       
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
    priors = np.stack(priors, axis=0)

    # features_out = np.zeros((sim_size,features.shape[1],features.shape[2],features.shape[3]))
    # edges_out = np.zeros((sim_size,gene_size,gene_size))

    # for i in range(sim_size):
    #     index = np.random.permutation(np.arange(experiment_size))
    #     num = np.random.permutation(np.arange(experiment_size))[0] 
    #     features_out[i,:,:,:] = features[num,:,:,:][:,:,index]
    #     edges_out[i,:,:] = edges[index,:][:,index]
    
    # Add noise
    # features_out = features_out + np.random.randn(sim_size,features.shape[1],features.shape[2],features.shape[3])
    # return features_out, edges_out
    return features, edges, priors

def convert_dataset_Dream4_sim_prior(feature_filename, edge_filename, sim_size, experiment_size=5, replic_size=5, gene_size=10, prior_strength=args.prior_strength):
    features = list()
    edges = list() 
    priors = list()

    for i in range(1, experiment_size+1):
        edge = read_edge_file(edge_filename+str(i)+".txt", gene_size=10)

        # Add prior
        prior_rand = np.random.rand(*edge.shape)
        prior = deepcopy(edge)
        index=np.where(prior_rand < prior_strength)

        # Change the groundtruth as the prior
        for j in np.arange(index[0].size):
	        prior[index[0][j],index[1][j]]=not edge[index[0][j],index[1][j]]

        for j in range(1, replic_size+1):
            features.append(read_feature_file(feature_filename+str(i)+"/insilico_size10_"+str(i)+"_timeseries_experiment"+str(j)+".tsv",gene_size=10,timestep_size=21))
            edges.append(edge)
            priors.append(prior)   
       
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
    priors = np.stack(priors, axis=0)

    features_out = np.zeros((sim_size,features.shape[1],features.shape[2],features.shape[3]))
    edges_out = np.zeros((sim_size,gene_size,gene_size))
    priors_out = np.zeros((sim_size,gene_size,gene_size))

    total_num = experiment_size * replic_size
    for i in range(sim_size):
        index = np.random.permutation(np.arange(gene_size))
        # num = np.random.permutation(np.arange(total_num))[0]
        num = i % total_num 
        features_out[i,:,:,:] = features[num,:,:,:][:,:,index]
        edges_out[i,:,:] = edges[num,:,:][index,:][:,index]
        priors_out[i,:,:] = priors[num,:,:][index,:][:,index]
    
    # Add noise
    # noise = 0.
    noise = 0.01
    features_out = features_out + np.random.randn(sim_size,features.shape[1],features.shape[2],features.shape[3])*noise
    return features_out, edges_out, priors_out

def convert_dataset_Dream4_sim_prior_train(feature_filename, edge_filename, noise_flag, start_replic, end_replic, sim_size,  experiment_size=5, replic_size=5,  gene_size=10, prior_strength=args.prior_strength):
    # replic_size =5, use 3 for training, 1 for validation, 1 for test
    features = list()
    edges = list() 
    priors = list()

    for i in range(1, experiment_size+1):
        edge = read_edge_file(edge_filename+str(i)+".txt", gene_size=10)

        # Add prior
        prior_rand = np.random.rand(*edge.shape)
        prior = deepcopy(edge)
        index=np.where(prior_rand < prior_strength)

        # Change the groundtruth as the prior
        for j in np.arange(index[0].size):
	        prior[index[0][j],index[1][j]]=not edge[index[0][j],index[1][j]]

        for j in range(start_replic, end_replic+1):
            features.append(read_feature_file(feature_filename+str(i)+"/insilico_size10_"+str(i)+"_timeseries_experiment"+str(j)+".tsv",gene_size=10,timestep_size=21))
            edges.append(edge)
            priors.append(prior)   
       
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
    priors = np.stack(priors, axis=0)

    features_out = np.zeros((sim_size,features.shape[1],features.shape[2],features.shape[3]))
    edges_out = np.zeros((sim_size,gene_size,gene_size))
    priors_out = np.zeros((sim_size,gene_size,gene_size))

    total_num = experiment_size * (end_replic-start_replic+1)
    
    for i in range(sim_size):
        index = np.random.permutation(np.arange(gene_size))
        # num = np.random.permutation(np.arange(total_num))[0]
        num = i % total_num 
        features_out[i,:,:,:] = features[num,:,:,:][:,:,index]
        edges_out[i,:,:] = edges[num,:,:][index,:][:,index]
        priors_out[i,:,:] = priors[num,:,:][index,:][:,index]
    
    # Add noise
    # noise = 0.
    noise = 0.01
    if noise_flag:
        features_out = features_out + np.random.randn(sim_size,features.shape[1],features.shape[2],features.shape[3])*noise
    return features_out, edges_out, priors_out



def convert_dataset_Dream4_sim(feature_filename, edge_filename, experiment_size=5, replic_size=5, gene_size=10, sim_size=5000):
    features = list()
    edges = list() 

    for i in range(1, experiment_size+1):
        edge = read_edge_file(edge_filename+str(i)+".txt", gene_size=10)
        for j in range(1, replic_size+1):
            features.append(read_feature_file(feature_filename+str(i)+"/insilico_size10_"+str(i)+"_timeseries_experiment"+str(j)+".tsv",gene_size=10,timestep_size=21))
            edges.append(edge)   
       
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    features_out = np.zeros((sim_size,features.shape[1],features.shape[2],features.shape[3]))
    edges_out = np.zeros((sim_size,gene_size,gene_size))

    total_num = experiment_size * replic_size
    for i in range(sim_size):
        index = np.random.permutation(np.arange(gene_size))
        # num = np.random.permutation(np.arange(total_num))[0]
        num = i % experiment_size 
        features_out[i,:,:,:] = features[num,:,:,:][:,:,index]
        edges_out[i,:,:] = edges[num,:,:][index,:][:,index]
    
    # Add noise
    features_out = features_out + np.random.randn(sim_size,features.shape[1],features.shape[2],features.shape[3])*0.1
    return features_out, edges_out

#features: loc + vel 
def convert_dataset_md(feature_filename, startIndex, experiment_size, timestep_size, feature_size, residue_size, interval):
    features = list()
    edges = list()

    for i in range(startIndex, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        features.append(read_feature_MD_file(feature_filename+"smd"+str(i)+".pdb", timestep_size, feature_size, residue_size, interval))
        edges.append(np.zeros((residue_size,residue_size)))
    
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
    
    return features, edges

#features: loc + vel 
#use sliding window to add 
def convert_dataset_md_single(feature_filename, startIndex, experiment_size, timestep_size, feature_size, residue_size, interval, window_start, window_end, aa_start, aa_end):
    features = list()
    edges = list()

    for i in range(startIndex, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        for j in range(window_start, window_end+1):
            print(str(i)+" "+str(j))
            features.append(read_feature_MD_file_slidingwindow(feature_filename+"ca_"+str(i)+".pdb", timestep_size, feature_size, residue_size, interval,j,aa_start,aa_end))
            edges.append(np.zeros((residue_size,residue_size)))
    print("***")
    print(len(features))
    print("###")
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
    
    return features, edges

#features: loc + vel + more
def convert_dataset_md_more(feature_filename, AAfeature_filename, experiment_size=20, timestep_size=20, feature_size=10, residue_size=47, interval=1000):
    features = list()
    edges = list()

    resDict = read_feature_Residue_file(AAfeature_filename)

    for i in range(1, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        features.append(read_feature_MD_file_resi(feature_filename+"smd"+str(i)+".pdb", resDict, timestep_size=50, feature_size=10, residue_size=47, interval=1000))
        edges.append(np.zeros((residue_size,residue_size)))
    
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)
   
    return features, edges


def timepoint_sim(feature,fold):
    # hard code now,fold=4
    # feature_shape: [timestep, feature_size, gene]
    step = 1/fold
    timestep = feature.shape[0]
    genes = feature.shape[2]   
    x=np.arange(timestep)
    xnew=np.arange(0, (timestep-1)+step, step)
    feature_out = np.zeros((xnew.shape[0],1,genes))
    for gene in range(genes):
        y = feature[:,0,gene]
        tck = interpolate.splrep(x, y, s=0)
        ynew = interpolate.splev(xnew, tck, der=0)
        feature_out[:,0,gene] = ynew
    return feature_out

feature_filename = "C:/Users/zhujx/Desktop/NRI/data/pdb/"
edge_filename    = ""

# All as the training

print("Generate Train")

features, edges = convert_dataset_md_single(feature_filename, startIndex=1, experiment_size=1, timestep_size=50, feature_size=6, residue_size=77, interval=60, window_start=1, window_end=56, aa_start=1, aa_end=77)

np.save('features.npy', features)
np.save('edges.npy', edges)


print("Generate Valid")

features_valid, edges_valid = convert_dataset_md_single(feature_filename, startIndex=1, experiment_size=1, timestep_size=50, feature_size=6, residue_size=77, interval=60, window_start=1, window_end=56, aa_start=1, aa_end=77)

np.save('features_valid.npy', features_valid)
np.save('edges_valid.npy', edges_valid)


print("Generate Test")

features_test, edges_test = convert_dataset_md_single(feature_filename, startIndex=1, experiment_size=1, timestep_size=50, feature_size=6, residue_size=77, interval=60, window_start=1, window_end=56, aa_start=1, aa_end=77)

np.save('features_test.npy', features_test)
np.save('edges_test.npy', edges_test)