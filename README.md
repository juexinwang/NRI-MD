# Neural relational inference to learn allosteric long-range interactions in proteins from molecular dynamics simulations

**Abstract:** Protein allostery is a biological process facilitated by spatially long-range intra-protein communication, whereby ligand binding or amino acid mutation at a distant site affects the active site remotely. Molecular dynamics (MD) simulation provides a powerful computational approach to probe the allostery effect. However, current MD simulations cannot reach the time scales of whole allostery processes. The advent of deep learning made it possible to evaluate both spatially short and long-range communications for understanding allostery. For this purpose, we applied a neural relational inference (NRI) model based on a graph neural network (GNN), which adopts an encoder-decoder architecture to simultaneously infer latent interactions to probe protein allosteric processes as dynamic networks of interacting residues. From the MD trajectories, this model successfully learned the long-range interactions and pathways that can mediate the allosteric communications between the two distant binding sites in the Pin1, SOD1, and MEK1 systems.

Please refer to the preprint. 

**Neural relational inference to learn allosteric long-range interactions in proteins from molecular dynamics simulations.**  
Jingxuan Zhu,  Juexin Wang, Weiwei Han,  Dong Xu 
https://www.biorxiv.org/content/10.1101/2021.01.20.427459v1

### Requirements
* Pytorch 1.2
* Python 3.7

### Prepare Molecular Simulation Trajectories

Place your molecular simulation trajectories in data/pdb folder. We have toy.pdb for the tutorial usage.

```
python generate_dataset.py
```

This step will seperate train/validation/test dataset, and generate .npy files in data folder.

### Run experiments

From the project's root folder, simply run
```
python main.py
```

Additionally, we provide code for an LSTM baseline (denoted *LSTM (joint)* in the paper), which you can run as follows:
```
python lstm_baseline.py
```

### Cite
If you make use of this code in your own work, please cite our paper:
```
@article{zhu2021neural,
  title={Neural relational inference to learn allosteric long-range interactions in proteins from molecular dynamics simulations},
  author={Zhu, Jingxuan and Wang, Juexin and Han, Weiwei and Xu, Dong},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Ref
Official implementation of neural relational inference
https://github.com/ethanfetaya/NRI

