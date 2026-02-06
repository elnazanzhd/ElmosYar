import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font for Persian support in plots
plt.rcParams['font.family'] = 'sans-serif' 

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_data(data):
    df = pd.DataFrame(data)
    
    # Expand evaluation_scores
    eval_scores_list = []
    for entry in data:
        scores = entry.get('evaluation_scores', {})
        if not scores:
            scores = {}
        eval_scores_list.append(scores)
    
    eval_df = pd.DataFrame(eval_scores_list)
    
    # Combine with main df
    df = pd.concat([df, eval_df], axis=1)
    
    # Map sentiment to numeric
    # POSITIVE -> 1, NEUTRAL -> 0, NEGATIVE -> -1
    # We will multiply by confidence score
    def get_sentiment_val(row):
        sentiment = row.get('bert_sentiment')
        score = row.get('bert_score')
        
        if pd.isna(score):
            return np.nan
            
        if sentiment == 'POSITIVE':
            return score
        elif sentiment == 'NEGATIVE':
            return -score
        else: # NEUTRAL
            return 0
            
    df['sentiment_numeric'] = df.apply(get_sentiment_val, axis=1)
    
    # Group by Professor
    # We want to aggregate:
    # 1. Mean of evaluation scores
    # 2. Mean of sentiment
    # 3. Count of comments (to filter out professors with too few comments if needed)
    
   
    score_cols = list(eval_df.columns)
    
    agg_dict = {col: 'mean' for col in score_cols}
    agg_dict['sentiment_numeric'] = 'mean'
    agg_dict['professor_name'] = 'count'
    grouped = df.groupby('professor_name').agg(agg_dict)
    grouped = grouped.rename(columns={'professor_name': 'comment_count'})
    
    # Fill NaN values with mean of column (for professors missing specific sub-scores)
    grouped = grouped.fillna(grouped.mean())
    
    return grouped, score_cols

def perform_clustering(df, feature_cols, k=4):

    X = df[feature_cols + ['sentiment_numeric']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    
    return df, X_scaled, kmeans

def visualize_clusters(df, X_scaled, kmeans):
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, 
        x='pca_1', 
        y='pca_2', 
        hue='cluster', 
        palette='viridis',
        s=100,
        alpha=0.8
    )
    
 
    
    plt.title('Professor Clusters (PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    
    output_path = 'professor_clusters_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def save_results(df, output_file=None):
    if output_file is None:
        output_file = os.path.join('data', 'output', 'professor_clusters.json')
    #
    output_df = df.reset_index()
    
    # Convert to list of dicts
    result = output_df.to_dict(orient='records')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

def main():
    input_file = os.path.join('data', 'processed', 'normalized_comments.json')
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print("Loading data...")
    data = load_data(input_file)
    
    print("Preprocessing and aggregating...")
    grouped_df, score_cols = preprocess_data(data)
    
    print(f"Found {len(grouped_df)} unique professors.")
    
    # Filter professors with very few comments?
    # grouped_df = grouped_df[grouped_df['comment_count'] > 1]
    
    print("Performing KMeans clustering...")
    k = 4
    clustered_df, X_scaled, model = perform_clustering(grouped_df, score_cols, k=k)
    
    print("Visualizing...")
    visualize_clusters(clustered_df, X_scaled, model)
    
    print("Saving results...")
    save_results(clustered_df)

    print("\nCluster Profiles (Mean Values):")
    profile = clustered_df.groupby('cluster')[score_cols + ['sentiment_numeric', 'comment_count']].mean()
    print(profile)

if __name__ == "__main__":
    main()
