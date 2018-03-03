"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODO(cathywu)s.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


# Vectorized function for hashing for np efficiency
def w(x): return np.int(hash(x)) % 1000
h = np.vectorize(w)


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        if y.size == 0:
            return 0
        p0 = np.where(y < 0.5)[0].size / y.size
        if np.abs(p0) < 1e-10 or np.abs(1-p0) < 1e-10:
            return 0
        return - p0 * np.log2(p0) - (1-p0) * np.log2(1-p0)

    @staticmethod
    def information_gain(X, y, thresh):
        base = DecisionTree.entropy(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        entropy = p0 * DecisionTree.entropy(y0) + p1 * DecisionTree.entropy(y1)
        return base - entropy

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:,idx] < thresh)[0]
        idx1 = np.where(X[:,idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([np.linspace(np.min(X[:, i]) + eps,
                                           np.max(X[:, i]) - eps, num=10) for i
                               in range(X.shape[1])])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in
                              thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains),
                                                   gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx,
                                        thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(max_depth=self.max_depth-1,
                                         feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(max_depth=self.max_depth-1,
                                          feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx,
                                                 thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (
                self.features[self.split_idx], self.thresh,
                self.left.__repr__(), self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params) for i in
            range(self.n)]

    def fit(self, X, y):
        for i in range(self.n):
            idx = np.random.randint(0, X.shape[0], X.shape[0])
            newX, newy = X[idx, :], y[idx]
            self.decision_trees[i].fit(newX, newy)
        return self

    def predict(self, X):
        yhat = [self.decision_trees[i].predict(X) for i in range(self.n)]
        return np.array(np.round(np.mean(yhat, axis=0)), dtype=np.bool)


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        i = 0
        while i < self.n:
            idx = np.random.choice(X.shape[0], size=X.shape[0], p=self.w)
            newX, newy = X[idx, :], y[idx]
            self.decision_trees[i].fit(newX, newy)
            wrong = np.abs((y - self.decision_trees[i].predict(X)))
            error = wrong.dot(self.w) / np.sum(self.w)
            self.a[i] = 0.5 * np.log2((1 - error) / error)
            # Update w
            wrong_idx = np.where(wrong > 0.5)[0]
            right_idx = np.where(wrong <= 0.5)[0]
            self.w[wrong_idx] = self.w[wrong_idx] * np.exp(self.a[i])
            self.w[right_idx] = self.w[right_idx] * np.exp(-self.a[i])
            self.w /= np.sum(self.w)
            i += 1
        return self

    def predict(self, X):
        yhat = [self.decision_trees[i].predict(X) for i in range(self.n)]
        p0 = self.a.dot(np.array(yhat) == 0)
        p1 = self.a.dot(np.array(yhat) == 1)
        return np.array(np.argmax(np.vstack([p0, p1]), axis=0), dtype=np.bool)


def preprocess(data, fill_mode=True, hash_cols=None, min_freq=10):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    back_map = None
    if hash_cols is not None:
        back_map = {}
        for col in hash_cols:
            counter = Counter(data[:, col])
            for term in counter.most_common():
                if term[-1] <= min_freq:
                    break
                else:
                    back_map[w(term[0])] = term[0]
            data[:, col] = h(data[:, col])
    data = np.array(data, dtype=np.float)

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(
                data[((data[:, i] < -1 - eps) + (data[:, i] > -1 + eps))][:,
                i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, back_map
    # return data, dict(zip(unique_hash, unique))


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in
                        counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        features = data[0, 1:]  # features = all columns except survived
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, back_map = preprocess(data[1:, 1:], hash_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:,:], hash_cols=[1, 5, 7, 8])
        print(back_map)

    elif dataset == "spam":
        features = ["pain", "private", "bank", "money", "drug", "spam",
                    "prescription", "creative", "height", "featured", "differ",
                    "width", "other", "energy", "business", "message",
                    "volumes", "revision", "path", "meter", "memo", "planning",
                    "pleased", "record", "out", "semicolon", "dollar", "sharp",
                    "exclamation", "parenthesis", "square_bracket", "ampersand"]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")

    # Basic decision tree
    print("\n\nPart (a, c): simplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    print("Tree structure", dt.__repr__())

    # Basic decision tree
    print("\n\nPart (a): sklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    from pydot import graph_from_dot_data
    import io
    out = io.StringIO()
    export_graphviz(clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # Bagged trees
    print("\n\nPart (d-e): bagged trees")
    bt = BaggedTrees(params, n=N)
    bt.fit(X, y)
    evaluate(bt)

    # Random forest
    print("\n\nPart (f-g): random forest")
    # rf = RandomForest(params, n=N, m=2)
    rf = RandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)

    # Boosted random forest
    print("\n\nPart (h): boosted random forest")
    boosted = BoostedRandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    boosted.fit(X, y)
    evaluate(boosted)

    if dataset == 'titanic':
        import sys; sys.exit()
    # The following is for the spam dataset only.
    # Sample code for determining which data are easier/harder to classify
    print("\n\nPart (i): easy/hard examples")
    A = np.argsort(boosted.w)
    tail = 1000
    bits = np.array([np.sum(X[A[i], :-7]) for i in range(X.shape[0])])
    print("Number of easiest %s samples containing >= 2 feature counts:" % tail,
          np.sum(bits[:tail] >= 2))
    print("Number of hardest %s samples containing < 2 feature counts:" % tail,
          np.sum(bits[-tail:] < 2))
    bits = np.array([np.sum(X[A[i], :]) for i in range(X.shape[0])])
    print("Number of easiest %s samples containing >= 5 feature counts:" % tail,
          np.sum(bits[:tail] >= 5))
    print("Number of hardest %s samples containing < 5 feature counts:" % tail,
          np.sum(bits[-tail:] < 5))
    idxes = 200-np.where(bits[-200:] > 7)[0]
    print("Most challenging samples: Examples among the 200 data most likely "
          "to be sampled which contained at least 7 feature counts")
    for idx in idxes:
        print('Example:', class_names[y[A[-idx]]],
              [x for x in zip(features, X[A[-idx], :]) if x[1] > 0],
              bits[-idx])
