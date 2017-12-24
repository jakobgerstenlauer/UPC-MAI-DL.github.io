import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn import decomposition, datasets
import sys
from matplotlib import pyplot as plt

if len(sys.argv) == 1:
    print 'usage: visualization.py fileName.npz itemsToVisualize'
    sys.exit()

#Read the name of the file with the embededding from the command line (it is the second argument, the first argument is the scriptname).
fileName=sys.argv[1]

# Load dataset embeddings and labels.
obj = np.load(fileName)
labels = obj['labels']
embeddings = obj['embeddings']

with open('Embedding'+fileName+'.csv','wb') as f:
    np.savetxt(f, embeddings, fmt='%.18f', delimiter=";")
f.close()

with open('Labels'+fileName+'.csv','wb') as g:
    np.savetxt(g, labels, fmt='%s', delimiter=";")
g.close()
