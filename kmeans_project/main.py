# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, rand_score, normalized_mutual_info_score, davies_bouldin_score

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def perform_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X), kmeans.cluster_centers_

def evaluate_clustering(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    rand_index = rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    db_index = davies_bouldin_score(X, y_pred)
    return f1, rand_index, nmi, db_index

def plot_clusters(X, y_pred, cluster_centers):
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', label='Centroids')
    plt.title("K-means Clustering on Iris Dataset")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y_true = load_data()
    y_pred, cluster_centers = perform_kmeans(X)
    f1, rand_index, nmi, db_index = evaluate_clustering(y_true, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Rand Index: {rand_index:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    plot_clusters(X, y_pred, cluster_centers)
