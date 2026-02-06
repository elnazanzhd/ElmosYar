import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_dataset(data, threshold=7.0):
    rows = []
    for entry in data:
        # Extract features
        description = entry.get('description', '')
        if not description:
            description = entry.get('raw_text', '') # Fallback
            
        # Skip if text is too short
        if not description or len(description.strip()) < 5:
            continue
            
        # Calculate Label (Average Score)
        scores = entry.get('evaluation_scores', {})
        if not scores:
            continue 
            
        avg_score = sum(scores.values()) / len(scores)
        label = 1 if avg_score >= threshold else 0
        
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
            'text': description,
            'sentiment_val': sent_val,
            'avg_score': avg_score,
            'label': label
        })
        
    return pd.DataFrame(rows)

def train_model(df):
    X = df[['text', 'sentiment_val']]
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution in Train: {y_train.value_counts(normalize=True).to_dict()}")
    
    # Pipeline
    # 1. Text -> TF-IDF
    # 2. Numeric -> Scaler
    # 3. Combine -> Logistic Regression
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), 'text'),
            ('num', StandardScaler(), ['sentiment_val'])
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
    ])
    
    print("\nTraining Logistic Regression Model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Good', 'Good']))
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return pipeline, X_test, y_test

def show_top_features(pipeline):
    # Get feature names
    preprocessor = pipeline.named_steps['preprocessor']
    tfidf = preprocessor.named_transformers_['text']
    feature_names = tfidf.get_feature_names_out()
    
    # Add numeric feature name
    feature_names = np.append(feature_names, 'sentiment_val')
    
    # Get coefficients
    classifier = pipeline.named_steps['classifier']
    coefs = classifier.coef_[0]
    
    # Sort
    top_positive = np.argsort(coefs)[-20:]
    top_negative = np.argsort(coefs)[:20]
    
    print("\nTop 20 Words indicating 'Good Experience':")
    print(", ".join([feature_names[i] for i in top_positive[::-1]]))
    
    print("\nTop 20 Words indicating 'Not Good':")
    print(", ".join([feature_names[i] for i in top_negative]))

def main():
    input_file = os.path.join('data', 'processed', 'normalized_comments.json')
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    data = load_data(input_file)
    df = prepare_dataset(data, threshold=7.0)
    
    print(f"Dataset prepared. Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    model, X_test, y_test = train_model(df)
    
    show_top_features(model)
    
    # Example predictions
    print("\nExample Predictions:")
    examples = X_test.sample(5, random_state=42)
    preds = model.predict(examples)
    probs = model.predict_proba(examples)[:, 1]
    
    for i, (idx, row) in enumerate(examples.iterrows()):
        original_row = df.loc[idx]
        
        print(f"\nText: {original_row['text'][:100]}...")
        print(f"True Label: {'Good' if original_row['label']==1 else 'Not Good'} (Score: {original_row['avg_score']:.1f})")
        print(f"Prediction: {'Good' if preds[i]==1 else 'Not Good'} (Prob: {probs[i]:.2f})")
        
    # Save model
    model_path = os.path.join('models', 'binary_classifier_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
