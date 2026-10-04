import numpy as np
import math
from utils.slidingWindows import find_length_rank


def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_PCA(train_data, test_data, periodicity=1, n_components=None, n_jobs=1):
    from models.PCA import PCA
    slidingWindow = find_length_rank(train_data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(train_data)
    score = clf.decision_function(test_data)
    return score.ravel()

def run_PCA(train_data, test_data, slidingWindow=100, n_components=None, n_jobs=1):
    from models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(train_data)
    score = clf.decision_function(test_data)
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1):
    from models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_SR(data, periodicity=1):
    from models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)
