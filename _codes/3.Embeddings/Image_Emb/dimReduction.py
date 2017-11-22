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
print embeddings.shape

if len(sys.argv) == 2:
    print 'Using all items for the dimensionality reduction.'
    print 'If you want to use only a subset of the first n labels: clustering.py fileName.npz n'
    _uniq = np.unique(labels)

if len(sys.argv) == 3:
    n_labels = int(sys.argv[2])
    _uniq = np.unique(labels)[:n_labels]

# Create the embeddings matrix (and its corresponding labels list)
# containing instances belonging to labels in the previous subset.
subsets_emb = np.zeros((0,4096))
true_labels = []
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    true_labels += [_lab for _ in range(part_emb.shape[0])]
true_labels = np.array(true_labels)

# Use a dimensionality reduction technique (e.g., PCA).
# Plot the PCA spectrum
pca = decomposition.PCA(n_components=30)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
reduced_emb = pca.fit(subsets_emb)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()

#Choose only the first 10 components
pca = decomposition.PCA(n_components=10)
reduced_emb = pca.fit_transform(subsets_emb)
np.savetxt("reduced_embedding_"+fileName+".csv", reduced_emb, delimiter=",")
