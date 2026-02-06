import json
import pandas as pd
import joblib
import os
import sys

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_inference_data(data):
    rows = []
    for entry in data:
        # Extract features
        description = entry.get('description', '')
        if not description:
            description = entry.get('raw_text', '') # Fallback
            
        # For inference, we even take short texts, but let's stick to meaningful ones
        if not description or len(description.strip()) < 2:
            description = " " # Empty string placeholder
            
        # Numeric Feature: Sentiment from BERT
        bert_sentiment = entry.get('bert_sentiment')
        bert_score = entry.get('bert_score', 0.0)
        
        if bert_sentiment == 'POSITIVE':
            sent_val = bert_score
        elif bert_sentiment == 'NEGATIVE':
            sent_val = -bert_score
        else:
            sent_val = 0.0
            
        rows.append({
            'professor_name': entry.get('professor_name', 'Unknown'),
            'text': description,
            'sentiment_val': sent_val
        })
        
    return pd.DataFrame(rows)

def main():
    model_path = os.path.join('models', 'binary_classifier_model.pkl')
    data_path = os.path.join('data', 'processed', 'normalized_comments.json')
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return
        
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading model and data...")
    model = joblib.load(model_path)
    data = load_data(data_path)
    
    print(f"Preparing data for {len(data)} comments...")
    df = prepare_inference_data(data)
    
    print("Running predictions...")
    # Predict Approval (1 = Good, 0 = Not Good)
    # We can also use predict_proba for a "Confidence Score"
    probabilities = model.predict_proba(df[['text', 'sentiment_val']])[:, 1]
    
    df['approval_prob'] = probabilities
    df['is_approved'] = (probabilities >= 0.5).astype(int)
    
    # Aggregate by Professor
    print("Aggregating results...")
    stats = df.groupby('professor_name').agg(
        total_comments=('text', 'count'),
        approved_count=('is_approved', 'sum'),
        avg_approval_score=('approval_prob', 'mean')
    ).reset_index()
    
    # Calculate Approval Rating (%)
    stats['approval_rating'] = (stats['approved_count'] / stats['total_comments']) * 100
    
    # Filter for significant results (e.g., at least 3 comments)
    significant_stats = stats[stats['total_comments'] >= 3].copy()
    
    # Sort by Approval Rating
    top_professors = significant_stats.sort_values(by=['approval_rating', 'total_comments'], ascending=[False, False])
    
    print("\n" + "="*50)
    print("🏆 TOP 10 APPROVED PROFESSORS (>= 3 comments)")
    print("="*50)
    print(top_professors.head(10)[['professor_name', 'approval_rating', 'total_comments']].to_string(index=False, float_format="%.1f"))
    
    print("\n" + "="*50)
    print("⚠️ LOWEST RATED PROFESSORS (>= 3 comments)")
    print("="*50)
    print(top_professors.tail(10)[['professor_name', 'approval_rating', 'total_comments']].to_string(index=False, float_format="%.1f"))
    
    # Save results
    output_file = os.path.join('data', 'professor_approval_ratings.csv')
    top_professors.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nFull leaderboard saved to {output_file}")

if __name__ == "__main__":
    main()
