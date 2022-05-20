import string
import numpy as np # linear algebra
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import nltk
import os, sys, email,re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

data = pd.read_csv('aapl.txt')

tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english', #tokenizer = tokenize_and_stem,
max_features = 20000)

tf_idf = tf_idf_vectorizor.fit_transform(data)

tf_idf_norm = normalize(tf_idf)

tf_idf_array = tf_idf_norm.toarray()

df1 = pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names())

class Kmeans:

    def __init__(self, k, seed = None, max_iter = 200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed = max_iter
        self.max_iter = max_iter

    def initialize_cetroids(self, data):
        initial_cetroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_cetroids]
        return self.centroids

    def assign_clusters(self, data):
        if data.ndim == 1:
            data == data.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(data, self.centroids, metric = 'euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)

        return self.cluster_labels

    def update_centroids(self, data):
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.k)])

        return self.centroids

    def convergence_calculation(self):
        pass

    def predict(self, data):
        return self.assign_clusters(data)

    def fit_kmeans(self, data):
        self.centroids = self.initialize_cetroids(data)

        for iter in range(self.max_iter):
            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            if iter % 100 == 0:
                print('running model iteration %d' %iter)
            print('model all done!')
            return self

# number_clusters = range(1, 7)
sklearn_pca = PCA(n_components= 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

# kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters ]

# score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]

# fitted = kmeans.fit(Y_sklearn)
# prediction = kmeans.predict(Y_sklearn)

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)
