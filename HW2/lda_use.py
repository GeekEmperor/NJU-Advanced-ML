from sklearn.decomposition import LatentDirichletAllocation
from util import load
import numpy as np

data, words = load()
for k in [5, 10, 20]:
    lda = LatentDirichletAllocation(n_components=k, random_state=0)
    lda.fit(data)
    components = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print(f'K = {k}')
    for i in range(k):
        indice = components.argsort(axis=1)[i, -10:]
        print(f'Topic {i}')
        for j in indice[::-1]:
            print((words[j], components[i, j]), end=' ')
        print()