import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


K = 3


"""ensemble methods"""


def balanced_random_forest(X, y, xtest):
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    brf.fit(X, y)
    y_pred = brf.predict(xtest)
    return y_pred


def easy_ensemble(X, y, xtest):
    eec = EasyEnsembleClassifier(random_state=0)
    eec.fit(X, y)
    y_pred = eec.predict(xtest)
    return y_pred


def bagging(X, y, xtest):
    bc = BaggingClassifier(base_estimator=KNeighborsClassifier(
        n_neighbors=3, algorithm='ball_tree'), random_state=0)
    bc.fit(X, y)
    y_pred = bc.predict(xtest)
    return y_pred


def balanced_bagging(X, y, xtest):
    bbc = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(
        n_neighbors=3, algorithm='ball_tree'),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=0)
    bbc.fit(X, y)
    y_pred = bbc.predict(xtest)
    return y_pred


def adaBoost(X, y, xtest):
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             n_estimators=100, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def rusBoost(X, y, xtest):
    rus_boost = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(),
        n_estimators=200, algorithm='SAMME.R', random_state=0)
    rus_boost.fit(X, y)
    y_pred = rus_boost.predict(xtest)
    return y_pred


"""classification and regression methods"""


def naive_bayes(X, y, xtest):
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(xtest)
    return y_pred


def decision_tree(X, y, xtest):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def svm_svc(X, y, xtest):
    clf = svm.SVC()
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def lr(X, y, xtest):
    l_r = LogisticRegression(random_state=0)
    l_r.fit(X, y)
    y_pred = l_r.predict(xtest)
    return y_pred


def random_forest(X, y, xtest):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(xtest)
    return y_pred


def nn(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


def nn_3(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


def nn_5(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


def nn_7(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


def nn_9(X, y, xtest):
    neigh = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree')
    neigh.fit(X, y)
    y_pred = neigh.predict(xtest)
    return y_pred


"""common methods"""


def call_knn(X, y, xtest):
    Ytest_pred = knn(K, X, y, xtest)
    return Ytest_pred


def origin_knn(X, y, xtest):
    return call_knn(X, y, xtest)


def knn(k, Xtrain, Ytrain, Xtest):
    """
    Classic kNN function. Take as input train features and labels. And
    test features. Then compute pairwise distances between test and train.
    And for each test example, return the majority class among its kNN.
    """
    # 计算训练集和测试集所有实例之间的相互距离
    d = euclidean_distances(Xtest, Xtrain, squared=True)
    # 找出距离最近的K个邻居的标签值
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtest.shape[0], k)  #
    #     # 找出最近的K个邻居中出现次数最多的标签值，作为预测结果
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    # X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
    #                           n_redundant=0, n_repeated=0, n_classes=3,
    #                           n_clusters_per_class=1,
    #                           weights=[0.01, 0.05, 0.94],
    #                           class_sep=0.8, random_state=0)
    X = np.array([[1, 2, 3, 4, 2, 1], [1, 2, 3, 4, 5, 1], [0, 1, 2, 3, 5, 0]])
    y = [1, 0, 1]
    x_test = np.array([[0, 1, 2, 3, 5, 4]])
    pred = random_over_sampler(X, y, x_test)
    print(pred)