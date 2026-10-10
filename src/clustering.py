"""
SmartCart Customer Segmentation - Clustering & Evaluation Engine
Author: Bhavesh Srivastava
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator


class CustomerSegmentationEngine:
    """
    Dimensionality reduction, hyperparameter selection, clustering model fitting,
    and statistical evaluation engine.
    """

    def __init__(self, n_pca_components: int = 3, random_state: int = 42):
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.kmeans = None
        self.agg = None
        self.optimal_k = None
        self.wcss = []
        self.silhouette_scores = []

    def fit_pca(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Projects standardized feature matrix into orthogonal principal components.
        """
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca

    def evaluate_cluster_range(self, X_pca: np.ndarray, k_min: int = 2, k_max: int = 10):
        """
        Calculates WCSS (Inertia) and Silhouette Scores across cluster range [k_min, k_max].
        """
        self.wcss = []
        for k in range(1, k_max + 1):
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
            km.fit(X_pca)
            self.wcss.append(km.inertia_)

        self.silhouette_scores = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X_pca)
            self.silhouette_scores.append(silhouette_score(X_pca, labels))

        # Detect Elbow point using KneeLocator
        knee = KneeLocator(range(1, k_max + 1), self.wcss, curve="convex", direction="decreasing")
        self.optimal_k = knee.knee or 4
        return self.optimal_k, self.wcss, self.silhouette_scores

    def fit_models(self, X_pca: np.ndarray, k: int = None):
        """
        Fits both K-Means and Agglomerative Clustering models and evaluates benchmarks.
        """
        if k is None:
            k = self.optimal_k or 4

        # 1. K-Means
        self.kmeans = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=self.random_state)
        kmeans_labels = self.kmeans.fit_predict(X_pca)

        # 2. Agglomerative Clustering
        self.agg = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        agg_labels = self.agg.fit_predict(X_pca)

        # 3. Model Benchmark Matrix
        benchmark_results = pd.DataFrame({
            "Algorithm": ["K-Means++", "Agglomerative (Ward)"],
            "Silhouette Score (Higher Better)": [
                silhouette_score(X_pca, kmeans_labels),
                silhouette_score(X_pca, agg_labels)
            ],
            "Davies-Bouldin Index (Lower Better)": [
                davies_bouldin_score(X_pca, kmeans_labels),
                davies_bouldin_score(X_pca, agg_labels)
            ],
            "Calinski-Harabasz Score (Higher Better)": [
                calinski_harabasz_score(X_pca, kmeans_labels),
                calinski_harabasz_score(X_pca, agg_labels)
            ]
        })

        return kmeans_labels, agg_labels, benchmark_results

    def generate_cluster_profiles(self, df_cleaned: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """
        Attaches cluster labels and computes mean centroids across all customer features.
        """
        df_profiled = df_cleaned.copy()
        df_profiled["Cluster"] = cluster_labels
        profiles = df_profiled.groupby("Cluster").mean()
        return profiles
