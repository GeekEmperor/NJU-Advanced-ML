import os
import numpy as np

def Load(dirname):
    labels = []
    features = []
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        with open(path) as f:
            label = f.readline().strip()
            label = [ord(c) - ord('a') for c in label]
            feature = []
            for line in f:
                line = line.replace(' ', ',')
                array = eval(line)
                feature.append(array)
            labels.append(np.array(label))
            features.append(np.array(feature, np.bool))
    return features, labels

def ForAndBack(feature, theta1, theta2, theta3):
    size = len(feature)
    Ms = np.zeros((size, 26, 26))
    Z = np.exp(theta1 + theta2 @ feature[0])
    Ms[0, 0] = Z
    for i in range(1, size):
        W = theta1 + theta2 @ feature[i] + theta3
        M = np.exp(W)
        Z = Z @ M
        Ms[i] = M
    Z = Z @ np.ones(26)
    alpha = np.zeros((size, 26))
    alpha[0] = Ms[0, 0]
    for i in range(1, size):
        alpha[i] = alpha[i - 1] @ Ms[i]
    beta = np.zeros((size, 26))
    beta[size - 1] = np.ones(26)
    for i in range(size - 2, -1, -1):
        beta[i] = Ms[i + 1] @ beta[i + 1]
    return Ms, Z, alpha, beta
    
# Train
trainX, trainY = Load('Dataset/train')
m = len(trainY)
f1 = np.zeros((m, 14, 26), np.bool)
f2 = np.zeros((m, 14, 26, 642), np.bool)
f3 = np.zeros((m, 14, 26, 26), np.bool)
for k in range(m):
    feature, label = trainX[k], trainY[k]
    for i, c in enumerate(label):
        f1[k, i, c] = 1
        f2[k, i, c, :321] = 1 - feature[i]
        f2[k, i, c, 321:] = feature[i]
    for i in range(len(label)-1):
        f3[k, i, label[i], label[i + 1]] = 1
g1 = f1.sum(axis=1)
g2 = f2.sum(axis=1)
g3 = f3.sum(axis=1)

theta1 = np.ones(26) * 0.1
theta2 = np.ones((26, 642)) * 0.1
theta3 = np.ones((26, 26)) * 0.1
lr = 0.001
for t in range(500):
    delta1 = np.zeros(26)
    delta2 = np.zeros((26, 642))
    delta3 = np.zeros((26, 26))

    for k in range(m):
        feature, label = trainX[k], trainY[k]
        feature = np.hstack((1 - feature, feature))
        l = len(feature)
        Ms, Z, alpha, beta = ForAndBack(feature, theta1, theta2, theta3)
        
        delta1 += g1[k]
        delta2 += g2[k]
        delta3 += g3[k]
        delta = np.sum(alpha * beta, axis=0) / Z
        delta1 -= delta
        for i in range(l):
            delta2 -= np.outer(delta, feature[i])
        for i in range(1, l):
            delta3 -= Ms[i] * np.outer(alpha[i - 1], beta[i]) / Z
    
    theta1 += lr * delta1 / m
    theta2 += lr * delta2 / m
    theta3 += lr * delta3 / m

# Test
testX, testY = Load('Dataset/test')
words = chars = 0
for feature, label in zip(testX, testY):
    feature = np.hstack((1 - feature, feature))
    l = len(feature)
    Ms, Z, alpha, beta = ForAndBack(feature, theta1, theta2, theta3)

    predict = np.argmax(alpha * beta, axis=1)
    words += np.all(predict == label)
    chars += np.sum(predict == label)
    predict = ''.join(chr(c) for c in predict + ord('a'))
    label =''.join(chr(c) for c in label + ord('a'))
    print(predict, label)

print('Acc of words:', words / len(testY))
print('Acc of chars:', chars / sum(len(label) for label in testY))