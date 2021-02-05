import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_roc_curve, roc_auc_score, accuracy_score

class TSVM:
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LinearSVC(loss='hinge', class_weight='balanced'))

    def fit(self, X_l, y_l, X_u):
        """
        训练函数
        :param X_l: 有标记数据的特征
        :param y: 有标记数据的标记
        :param X_u: 无标记数据的特征
        """
        y_l[y_l == 0] = -1
        self.clf.fit(X_l, y_l)
        y_u = self.clf.predict(X_u)
        c_l = 1
        c_u = 1e-3
        X = np.vstack([X_l, X_u])
        y = np.concatenate([y_l, y_u])
        t = np.arange(y_u.shape[0])
        sample_weight = np.ones(y_l.shape[0] + y_u.shape[0])
        sample_weight[y_l.shape[0]:] = c_u
        while c_u < c_l:
            while True:
                self.clf.fit(X, y, linearsvc__sample_weight=sample_weight)
                xi = 1 - y_u * self.clf.decision_function(X_u)
                i = t[y_u > 0][np.argmax(xi[y_u > 0])]
                j = t[y_u < 0][np.argmax(xi[y_u < 0])]
                if xi[i] > 0 and xi[j] > 0 and xi[i] + xi[j] > 2:
                    y_u[i] = -y_u[i]
                    y_u[j] = -y_u[j]
                    y = np.concatenate([y_l, y_u])
                else:
                    break
            c_u = np.min([2*c_u, c_l])
            sample_weight[y_l.shape[0]:] = c_u
        self._estimator_type = self.clf._estimator_type
        self.classes_ = [0, 1]
        y_l[y_l == -1] = 0

    def predict(self, X):
        """
        预测函数
        :param X: 预测数据的特征
        :return: 数据对应的预测值
        """
        y = self.clf.predict(X)
        y[y < 0] = 0
        return y

    def decision_function(self, X):
        return self.clf.decision_function(X)


def load_data():
    label_X = np.loadtxt('label_X.csv', delimiter=',')
    label_y = np.loadtxt('label_y.csv', delimiter=',').astype(np.int)
    unlabel_X = np.loadtxt('unlabel_X.csv', delimiter=',')
    unlabel_y = np.loadtxt('unlabel_y.csv', delimiter=',').astype(np.int)
    test_X = np.loadtxt('test_X.csv', delimiter=',')
    test_y = np.loadtxt('test_y.csv', delimiter=',').astype(np.int)
    return label_X, label_y, unlabel_X, unlabel_y, test_X, test_y


if __name__ == '__main__':
    label_X, label_y, unlabel_X, unlabel_y, test_X, test_y \
        = load_data()
    tsvm = TSVM()
    tsvm.fit(label_X, label_y, unlabel_X)
    print('Acc On Label: ', accuracy_score(label_y, tsvm.predict(label_X)))
    print('Acc On Unlabel: ', accuracy_score(unlabel_y, tsvm.predict(unlabel_X)))
    print('Acc On Test: ', accuracy_score(test_y, tsvm.predict(test_X)))
    print('Auc On Label: ', roc_auc_score(label_y, tsvm.decision_function(label_X)))
    print('Auc On Unlabel: ', roc_auc_score(unlabel_y, tsvm.decision_function(unlabel_X)))
    print('Auc On Test: ', roc_auc_score(test_y, tsvm.decision_function(test_X)))
    fig, ax = plt.subplots()
    plot_roc_curve(tsvm, label_X, label_y, name='Label', ax=ax)
    plot_roc_curve(tsvm, unlabel_X, unlabel_y, name='Unlabel', ax=ax)
    plot_roc_curve(tsvm, test_X, test_y, name='Test', ax=ax)
    plt.show()
    fig.savefig('Preprocess_Balanced.png')