# Databricks notebook source
#'-U' upgrades the package to the latest available version
%pip install scikit-learn 
%pip install pandas 
%pip install numpy

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import pandas_udf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# COMMAND ----------

"""Load contactos data for classical clustering"""
def load_and_preprocess_gold_contactos():
    df = spark.table("workspace.sc_gold.contactos_pbs")
    # Convert to Pandas for easier processing
    df_pd = df.toPandas()
    
    return df_pd

# COMMAND ----------

"""Prepare features for classical clustering using numerical and categorical data"""
def prepare_classical_features(df):
    # Select relevant features for clustering
    feature_columns = [
        'origem', 'formulario', 'tipo_de_pedido', 'modelo', 
        'consentimento', 'email_opt_out', 'agrupamento_cliente', 
        'caracterizacao'
    ]
    
    # Create a copy with only the features we need
    df_features = df[feature_columns].copy()
    
    # Handle missing values by filling with 'unknown'
    df_features = df_features.fillna('unknown')

    # Convert email_opt_out to numeric (binary encoding)
    df_features['email_opt_out'] = df_features['email_opt_out'].map({'Não': 0, 
                                                                     'Sim': 1})
    # Handle any remaining NaN values in email_opt_out (if 'unknown' was filled)
    df_features['email_opt_out'] = df_features['email_opt_out'].fillna(0)
    
    # Identify categorical and numerical columns
    categorical_columns = [
        'origem', 'formulario', 'tipo_de_pedido', 'modelo',
        'consentimento', 'agrupamento_cliente', 'caracterizacao'
    ]
    
    numerical_columns = ['email_opt_out']  # Add any numerical columns if they exist in your data
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns),
            ('num', StandardScaler(), numerical_columns) if numerical_columns else ('num', 'passthrough', [])
        ],
        remainder='drop'
    )
    
    # Fit and transform the data
    print(f"Preprocessing {len(df_features)} contactos with classical features...")
    
    features_encoded = preprocessor.fit_transform(df_features)
    
    print(f"Generated {features_encoded.shape[1]} features from categorical variables")
    
    return features_encoded, preprocessor, df_features

# COMMAND ----------

"""Perform K-means clustering with optimal cluster selection"""
def perform_clustering(features, n_clusters_range=range(4, 5)):
    best_score = -1
    best_k = 4
    scores = []
    
    # Find optimal number of clusters using the silhouette score
    print("Finding optimal number of clusters...")
    for k in n_clusters_range:
        # n_init=10 runs 10 times with different centroids and picks the best result
        kmeans = KMeans(n_clusters=k, 
                        random_state=42, 
                        n_init=10,
                        max_iter=300)
        # Assigns each data point to a cluster
        cluster_labels = kmeans.fit_predict(features)
        # silhouette_score measures how well each point fits within its cluster
        score = silhouette_score(features, cluster_labels)
        scores.append(score)
        
        print(f"K={k}: Silhouette Score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final clustering with best k
    final_kmeans = KMeans(n_clusters=best_k, 
                          random_state=42, 
                          n_init=10,
                          max_iter=300)
    final_labels = final_kmeans.fit_predict(features)
    
    return final_labels, best_k, best_score, scores, final_kmeans

# COMMAND ----------

"""Analyze cluster characteristics"""
def analyze_clusters(df, cluster_labels):
    df_clustered = df.copy()
    df_clustered['cluster_labels'] = cluster_labels
    
    cluster_analysis = {}
    # Analyze cluster distributions
    for cluster_label in range(len(set(cluster_labels))):
        cluster_data = df_clustered[df_clustered['cluster_labels'] == cluster_label]
        
        analysis = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100,
            'top_origem': cluster_data['origem'].value_counts().head(3).to_dict(),
            'top_modelo': cluster_data['modelo'].value_counts().head(3).to_dict(),
            'top_formulario': cluster_data['formulario'].value_counts().head(3).to_dict(),
            'top_tipo_pedido': cluster_data['tipo_de_pedido'].value_counts().head(3).to_dict(),
            'top_consentimento': cluster_data['consentimento'].value_counts().head(3).to_dict(),
            'top_agrupamento': cluster_data['agrupamento_cliente'].value_counts().head(3).to_dict(),
            'top_caracterizacao': cluster_data['caracterizacao'].value_counts().head(3).to_dict(),
            'email_opt_out_stats': {
                'mean': cluster_data['email_opt_out'].mean(),
                'percentage_opt_out': (cluster_data['email_opt_out'] == 1).sum() / len(cluster_data) * 100,
                'count_sim': (cluster_data['email_opt_out'] == 1).sum(),
                'count_nao': (cluster_data['email_opt_out'] == 0).sum()
            }
        }

        cluster_analysis[f'Cluster_{cluster_label}'] = analysis
    
    return df_clustered, cluster_analysis

# COMMAND ----------

"""Create visualizations for cluster analysis"""
def visualize_clusters(features, cluster_labels, best_k):    
    # Principal Component Analysis (PCA) reduces high-dimensional data down to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    scatter = axes[0].scatter(features_2d[:, 0], 
                              features_2d[:, 1], 
                              c=cluster_labels, 
                              cmap='tab10', 
                              alpha=0.6)
    axes[0].set_title(f'Contacto - Clusters (PCA) - {best_k} clusters')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0])
    
    # Cluster distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    axes[1].bar(unique_labels, counts)
    axes[1].set_title('Cluster Distribution')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Contactos')
    
    plt.tight_layout()
    plt.show()

# COMMAND ----------

"""Create visualizations for cluster analysis, excluding outlier clusters"""
def visualize_clusters_optimized(features, cluster_labels, best_k, outliers_to_remove=3):
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Identify the smallest clusters (potential outliers)
    smallest_indices = np.argsort(counts)[:outliers_to_remove]
    smallest_cluster_ids = unique_labels[smallest_indices]
    
    # Filter out these outlier clusters
    mask = ~np.isin(cluster_labels, smallest_cluster_ids)
    filtered_features = features[mask]
    filtered_labels = cluster_labels[mask]
    
    # Update remaining cluster labels to be contiguous (0, 1, 2, ...)
    label_mapping = {old_label: new_label for new_label, old_label in 
                    enumerate(np.unique(filtered_labels))}
    mapped_labels = np.array([label_mapping[label] for label in filtered_labels])
    
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(filtered_features)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    scatter = axes[0].scatter(features_2d[:, 0], 
                              features_2d[:, 1], 
                              c=mapped_labels, 
                              cmap='tab10', 
                              alpha=0.6)
    axes[0].set_title(f'Contacto - Clusters (PCA) - {best_k - outliers_to_remove} clusters (outliers removed)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0])
    
    # Cluster distribution
    filtered_unique_labels, filtered_counts = np.unique(mapped_labels, return_counts=True)
    axes[1].bar(filtered_unique_labels, filtered_counts)
    axes[1].set_title('Cluster Distribution (Outliers Removed)')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Contactos')
       
    plt.tight_layout()
    plt.show()
    
    return filtered_features, mapped_labels

# COMMAND ----------

"""Analyze feature importance for clustering"""
def analyze_feature_importance(kmeans_model, preprocessor, original_features):
    """
    Analyze which features are most important for the clustering
    by examining cluster centers
    """
    cluster_centers = kmeans_model.cluster_centers_
    
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback for older sklearn versions or OneHotEncoder only
        cat_features = []
        for col in original_features.columns:
            unique_vals = original_features[col].unique()
            # OneHot drops first category, so we skip it
            for val in unique_vals[1:]:
                cat_features.append(f"{col}_{val}")
        feature_names = cat_features
    
    # Calculate feature importance as the variance of cluster centers
    feature_importance = np.var(cluster_centers, axis=0)
    
    # Sort features by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

# COMMAND ----------

"""Main function to execute the classical clustering pipeline"""

print("Loading contactos data...")
contacto_df = load_and_preprocess_gold_contactos()

print("Preparing classical features...")
features, preprocessor, original_features = prepare_classical_features(contacto_df)

print("Performing clustering...")
cluster_labels, best_k, best_score, scores, kmeans_model = perform_clustering(features)

print(f"Optimal number of clusters: {best_k}")
print(f"Best silhouette score: {best_score:.3f}")

# COMMAND ----------

print("Performing clustering...")
cluster_labels, best_k, best_score, scores, kmeans_model = perform_clustering(features)

print(f"Optimal number of clusters: {best_k}")
print(f"Best silhouette score: {best_score:.3f}")

# COMMAND ----------

print("Analyzing clusters...")
clustered_df, cluster_analysis = analyze_clusters(original_features, cluster_labels)

# Print cluster analysis
for cluster_name, analysis in cluster_analysis.items():
    print(f"\n{cluster_name}:")
    print(f"  Size: {analysis['size']} ({analysis['percentage']:.1f}%)")
    print(f"  Top Origins: {analysis['top_origem']}")
    print(f"  Top Models: {analysis['top_modelo']}")
    print(f"  Top Formularios: {analysis['top_formulario']}")
    print(f"  Top Tipo Pedido: {analysis['top_tipo_pedido']}")
    print(f"  Top Consentimento: {analysis['top_consentimento']}")
    print(f"  Top Agrupamento: {analysis['top_agrupamento']}")
    print(f"  Top Caracterizacao: {analysis['top_caracterizacao']}")
    print(f"  Email Opt-out: {analysis['email_opt_out_stats']['percentage_opt_out']:.1f}% opt-out ({analysis['email_opt_out_stats']['count_sim']} Sim, {analysis['email_opt_out_stats']['count_nao']} Não)")

# Analyze feature importance
print("Analyzing feature importance...")
feature_importance = analyze_feature_importance(kmeans_model, preprocessor, original_features)
print("\nTop 10 most important features for clustering:")
print(feature_importance.head(10))

# Visualize results
print("Creating visualizations...")
visualize_clusters(features, cluster_labels, best_k)

# COMMAND ----------

print("Analyzing 4 clusters...")
clustered_df, cluster_analysis = analyze_clusters(original_features, cluster_labels)

# Print cluster analysis
for cluster_name, analysis in cluster_analysis.items():
    print(f"\n{cluster_name}:")
    print(f"  Size: {analysis['size']} ({analysis['percentage']:.1f}%)")
    print(f"  Top Origins: {analysis['top_origem']}")
    print(f"  Top Models: {analysis['top_modelo']}")
    print(f"  Top Formularios: {analysis['top_formulario']}")
    print(f"  Top Tipo Pedido: {analysis['top_tipo_pedido']}")
    print(f"  Top Consentimento: {analysis['top_consentimento']}")
    print(f"  Top Agrupamento: {analysis['top_agrupamento']}")
    print(f"  Top Caracterizacao: {analysis['top_caracterizacao']}")
    print(f"  Email Opt-out: {analysis['email_opt_out_stats']['percentage_opt_out']:.1f}% opt-out ({analysis['email_opt_out_stats']['count_sim']} Sim, {analysis['email_opt_out_stats']['count_nao']} Não)")

# Analyze feature importance
print("Analyzing feature importance...")
feature_importance = analyze_feature_importance(kmeans_model, preprocessor, original_features)
print("\nTop 10 most important features for clustering:")
print(feature_importance.head(10))

# Visualize results
print("Creating visualizations...")
visualize_clusters(features, cluster_labels, best_k)

# COMMAND ----------

# Visualize results with outliers removed
print("Creating optimized visualizations (removing outlier clusters)...")
filtered_features, filtered_labels = visualize_clusters_optimized(features, cluster_labels, best_k, outliers_to_remove=1)

# COMMAND ----------

# Visualize results for different numbers of outliers removed
visualize_clusters_optimized(features, cluster_labels, best_k, outliers_to_remove=2)

# COMMAND ----------

# Visualize results for more outliers removed
visualize_clusters_optimized(features, cluster_labels, best_k, outliers_to_remove=3)

# COMMAND ----------

"""Create a heatmap to show cluster characteristics"""
def create_cluster_heatmap(clustered_df):
    # Select categorical columns for analysis
    categorical_cols = ['origem', 'formulario', 'tipo_de_pedido', 'modelo', 
                       'consentimento', 'agrupamento_cliente', 'caracterizacao']
    
    # Create a summary of each cluster's characteristics
    cluster_summary = []
    
    for cluster_id in sorted(clustered_df['cluster_labels'].unique()):
        cluster_data = clustered_df[clustered_df['cluster_labels'] == cluster_id]
        
        row = {'cluster': f'Cluster_{cluster_id}', 'size': len(cluster_data)}
        
        # For each categorical column, find the most common value and its percentage
        for col in categorical_cols:
            most_common = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else 'unknown'
            percentage = (cluster_data[col] == most_common).sum() / len(cluster_data) * 100
            row[f'{col}_dominant'] = most_common
            row[f'{col}_pct'] = percentage

        # Handle email_opt_out as numerical feature
        row['email_opt_out_pct'] = (cluster_data['email_opt_out'] == 1).sum() / len(cluster_data) * 100

        cluster_summary.append(row)
    
    summary_df = pd.DataFrame(cluster_summary)
    
    # Create heatmap of percentages
    pct_cols = [col for col in summary_df.columns if col.endswith('_pct')]
    heatmap_data = summary_df[['cluster'] + pct_cols].set_index('cluster')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'})
    plt.title('Cluster Characteristics Heatmap\n(Percentage of dominant category in each cluster)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return summary_df

# COMMAND ----------

# Create cluster characteristics heatmap
print("Creating cluster characteristics heatmap...")
cluster_summary = create_cluster_heatmap(clustered_df)
print("\nCluster Summary:")
print(cluster_summary.round(2))
