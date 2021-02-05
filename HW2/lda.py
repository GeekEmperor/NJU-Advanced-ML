from scipy.sparse.construct import rand
from util import load
import numpy as np

class LDA:
    def __init__(self, K=5, iter=10):
        self.K = 5
        self.iter = iter
    
    def sample(self, t, n):
        w = self.words[t][n]
        old = self.z[t][n]
        self.count1[old, w] -= 1
        self.count2[t, old] -= 1

        p = np.zeros(self.K)
        for k in range(self.K):
            p[k] = (self.count2[t, k] + self.alpha[k]) * (self.count1[k, w] + self.eta[w]) /\
                (np.sum(self.count1[k]) + np.sum(self.eta)) ** 2
        p /= p.sum()

        new = np.argmax(np.random.multinomial(1, p))
        self.count1[new, w] += 1
        self.count2[t, new] += 1
        self.z[t][n] = new

    def fit(self, data):
        self.T, self.N = data.shape
        self.alpha = np.ones(self.K)
        self.theta = np.ones((self.T, self.K))
        self.eta = np.ones(self.N)
        self.beta = np.ones((self.K, self.N))
        self.count1 = np.zeros((self.K, self.N))
        self.count2 = np.zeros((self.T, self.K))
        self.words = np.array([np.zeros(data[t].sum(), dtype=int) for t in range(self.T)], dtype=object)
        for t in range(self.T):
            n = 0
            for w, c in enumerate(data[t]):
                for i in range(c):
                    self.words[t][n] = w
                    n += 1
        self.z = np.array([np.zeros(data[t].sum(), dtype=int) for t in range(self.T)], dtype=object)
        for t in range(self.T):
            for n in range(len(self.z[t])):
                k = np.random.randint(0, self.K)
                w = self.words[t][n]
                self.z[t][n] = k
                self.count1[k, w] += 1
                self.count2[t, k] += 1
        
        for i in range(self.iter):
            for t in range(self.T):
                for n in range(len(self.words[t])):
                    self.sample(t, n)
        
        for k in range(self.K):
            for n in range(self.N):
                self.beta[k, n] = self.count1[k, n] + self.eta[n]
        self.beta = self.beta / self.beta.sum(axis=1, keepdims=True)
    
    def prob(self):
        return self.beta

data, words = load()
for k in [5, 10, 20]:
    lda = LDA(k, 10)
    lda.fit(data)
    probs = lda.prob()
    print(f'K = {k}')
    for i in range(k):
        indice = probs.argsort(axis=1)[i, -10:]
        print(f'Topic {i}')
        for j in indice[::-1]:
            print((words[j], probs[i, j]), end=' ')
        print()
