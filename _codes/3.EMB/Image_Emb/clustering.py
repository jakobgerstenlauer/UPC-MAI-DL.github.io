import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA

obj = np.load('mit67_embeddings.npz')
labels = obj['labels']
embeddings = obj['embeddings']

n_labels = 5
_uniq = np.unique(labels)[:n_labels]
print 'The {n} labels selected to visualize:'.format(n=n_labels), _uniq

subsets_emb = np.zeros((0,4096))
true_labels = []
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    true_labels += [_lab for _ in range(part_emb.shape[0])]
true_labels = np.array(true_labels)

reduced_emb = PCA(n_components=100).fit_transform(subsets_emb)

pred_labels = KMeans(n_clusters=n_labels, random_state=0).fit_predict(reduced_emb)
nmi = NMI(true_labels, pred_labels)
print 'NMI score:', nmi