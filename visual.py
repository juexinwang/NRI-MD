import numpy as np
a = np.load('056_500probs_test.npy')
b = a[:, :, 1]
c = a[:, :, 2]
d = a[:, :, 3]
probs = b+c+d
probs = np.reshape(probs, (56,5852))
edges_train= probs/56

results=np.zeros((5852))
for i in range(56):
      results=results+edges_train[i,:]

index=results<(0.5)
results[index]= 0


genes=77
edges_results=np.zeros((genes,genes))
count=0
for i in range(genes):
      for j in range(genes):
              if not i==j:
                      edges_results[i,j]=results[count]
                      count+=1
              else:
                      edges_results[i,j]=0

import matplotlib.pyplot as plt

import seaborn as sns
a = edges_results
ax = sns.heatmap(a, linewidth=0.5, cmap="Blues", vmax=1.0, vmin=0.0)
plt.savefig('probs.png', dpi=600)
plt.show()

