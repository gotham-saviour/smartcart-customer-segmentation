"""
SmartCart Customer Segmentation - CLI Pipeline Driver
Author: Bhavesh Srivastava
"""

import os
import sys
import argparse
import pandas as pd

# Add current directory to path for robust imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import SmartCartPreprocessor
from clustering import CustomerSegmentationEngine


def run_pipeline(data_path: str, output_dir: str):
    print("=" * 65)
    print(" SmartCart Customer Segmentation Pipeline")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data file not found at: {data_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/5] Ingesting customer data from '{data_path}'...")
    df_raw = pd.read_csv(data_path)
    print(f"      Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} raw attributes.")

    # 2. Preprocess & Feature Engineer
    print("\n[2/5] Executing domain-driven feature engineering and scaling...")
    preprocessor = SmartCartPreprocessor()
    X_scaled, df_cleaned = preprocessor.fit_transform(df_raw)
    print(f"      Engineered {df_cleaned.shape[1]} clean attributes for {len(df_cleaned)} valid customers.")

    # 3. PCA Dimensionality Reduction
    print("\n[3/5] Performing Principal Component Analysis (3 components)...")
    engine = CustomerSegmentationEngine(n_pca_components=3)
    X_pca = engine.fit_pca(X_scaled)
    var_ratio = engine.pca.explained_variance_ratio_
    print(f"      PCA Variance Captured: {var_ratio.sum() * 100:.2f}% (PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%, PC3: {var_ratio[2]*100:.1f}%)")

    # 4. Cluster Selection & Fitting
    print("\n[4/5] Evaluating optimal K (Elbow Method & Silhouette Analysis)...")
    optimal_k, wcss, sil_scores = engine.evaluate_cluster_range(X_pca)
    print(f"      Optimal K determined: {optimal_k}")

    print(f"\n      Benchmarking K-Means++ vs. Agglomerative Clustering (K={optimal_k})...")
    kmeans_labels, agg_labels, benchmarks = engine.fit_models(X_pca, k=optimal_k)
    print("\n" + benchmarks.to_string(index=False))

    # 5. Export Profiled Output
    print("\n[5/5] Generating customer cluster assignments and strategic profiles...")
    df_segmented = df_cleaned.copy()
    df_segmented["Cluster"] = kmeans_labels

    out_csv = os.path.join(output_dir, "segmented_customers.csv")
    df_segmented.to_csv(out_csv, index=False)
    print(f"      Saved segmented customers dataset to: {out_csv}")

    profiles = engine.generate_cluster_profiles(df_cleaned, kmeans_labels)
    out_profiles = os.path.join(output_dir, "cluster_centroid_summary.csv")
    profiles.to_csv(out_profiles)
    print(f"      Saved cluster centroid profiles to: {out_profiles}")

    print("\n" + "=" * 65)
    print(" [SUCCESS] Segmentation Pipeline Execution Complete!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartCart Customer Segmentation Pipeline")
    parser.add_argument("--data", type=str, default="data/smartcart_customers.csv", help="Path to input CSV")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    # Fallback to local file if run from root
    if not os.path.exists(args.data) and os.path.exists("smartcart_customers.csv"):
        args.data = "smartcart_customers.csv"

    run_pipeline(args.data, args.out)
