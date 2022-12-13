######################################
# Deployed in the HPC backend
######################################
import numpy as np
import argparse,os

parser = argparse.ArgumentParser('Preprocessing: Generate training/validation/testing features from pdb in web server')
parser.add_argument('--MDfolder', type=str, default="/N/u/soicwang/BigRed200/inputPDBDir/",
                    help='folder of pdb MD')
parser.add_argument('--inputFile', type=str, default="1213AAAA.pdb",
                    help='inputFile name')
parser.add_argument('--datafolder', type=str, default="/N/u/soicwang/BigRed200/inputPDBDir/1213AAAA/data/",
                    help='folder store the preprocessed data')
parser.add_argument('--start', type=int, default="1",
                    help='select start residue')
parser.add_argument('--end', type=int, default="56",
                    help='select end residue')
parser.add_argument('--timestep-size', type=int, default=50,
                    help='timestep size, very important')

parser.add_argument('--train-interval', type=int, default=60,
                    help='intervals in trajectory in training')
parser.add_argument('--validate-interval', type=int, default=80,
                    help='intervals in trajectory in validate')
parser.add_argument('--test-interval', type=int, default=100,
                    help='intervals in trajectory in test')
args = parser.parse_args()


def read_feature_MD_file_slidingwindow(filename, timestep_size, num_residues, interval, window_choose, aa_start, aa_end):
    """read single expriments of all time points
    """
    feature = np.zeros((timestep_size, 6, num_residues))

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


def convert_dataset_md_single(MDfolder, inputFile, timestep_size, num_residues, interval, resi_start, resi_end, aa_start, aa_end):
    """
    Convert in single md file in single skeleton
    """
    features = list()
    edges = list()

    for j in range(resi_start, resi_end+1):
        # print(str(i)+" "+str(j))
        features.append(read_feature_MD_file_slidingwindow(MDfolder+inputFile, timestep_size, num_residues, interval, j, aa_start, aa_end))
        edges.append(np.zeros((num_residues, num_residues)))
    
    print("***")
    print(len(features))
    print("###")
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    return features, edges

def validate_param(MDfolder, inputFile):
    '''Validate params, should work in the front end'''

    num_residues = 100000
    oriResiNum = -1
    totalModelNum = 0

    with open(MDfolder+inputFile) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if(line.startswith("MODEL")):
                modelNum = int(words[1])
                totalModelNum = modelNum
                if modelNum == 1:
                    # Do nothing
                    pass                    
                else:
                    oriResiNum = -1
                    if modelNum == 2:
                        num_residues = resiNum
                    else:
                        if not resiNum == num_residues:
                            print('Model error exists in the input:'+str(modelNum))
            elif(line.startswith("ATOM") and words[2] == "CA"):
                resiNum = int(words[1])
                if resiNum > num_residues or resiNum <= oriResiNum:
                    print('Residue error exists in the input model:'+str(modelNum))
                    return
                else:
                    oriResiNum = resiNum
    return num_residues, totalModelNum

                

feature_size = 6
MDfolder = args.MDfolder
inputFile = args.inputFile
datafolder = args.datafolder

resi_start = args.start
resi_end = args.end
timestep_size = args.timestep_size

## Validate
num_residues, totalModelNum = validate_param(MDfolder, inputFile)
print('Num of residues:'+str(num_residues)+'\t Num of models:'+str(totalModelNum))

# Check and validate the split of training/validation/test
train_interval = args.train_interval
validate_interval = args.validate_interval
test_interval = args.test_interval

if train_interval*timestep_size<totalModelNum:
    train_interval=int(np.ceil(totalModelNum/timestep_size))
if validate_interval*timestep_size<totalModelNum:
    validate_interval=int(np.ceil(totalModelNum/timestep_size))
if test_interval*timestep_size<totalModelNum:
    test_interval=int(np.ceil(totalModelNum/timestep_size))

# check and Generate folder
if not os.path.isdir(datafolder):
    os.makedirs(datafolder)

# Generate training/validating/testing
print("Generate Train")
features, edges = convert_dataset_md_single(MDfolder, inputFile, timestep_size=timestep_size,
                                            num_residues=num_residues, interval=train_interval, resi_start=resi_start, resi_end=resi_end, aa_start=1, aa_end=num_residues)

np.save(datafolder+'features.npy', features)
np.save(datafolder+'edges.npy', edges)


print("Generate Valid")
features_valid, edges_valid = convert_dataset_md_single(MDfolder, inputFile, timestep_size=timestep_size,
                                                        num_residues=num_residues, interval=validate_interval, resi_start=resi_start, resi_end=resi_end, aa_start=1, aa_end=num_residues)

np.save(datafolder+'features_valid.npy', features_valid)
np.save(datafolder+'edges_valid.npy', edges_valid)


print("Generate Test")
features_test, edges_test = convert_dataset_md_single(MDfolder, inputFile, timestep_size=timestep_size,
                                                      num_residues=num_residues, interval=test_interval, resi_start=resi_start, resi_end=resi_end, aa_start=1, aa_end=num_residues)
np.save(datafolder+'features_test.npy', features_test)
np.save(datafolder+'edges_test.npy', edges_test)