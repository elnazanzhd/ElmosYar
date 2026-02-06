import json
import torch
import os
from transformers import pipeline

def main():
    input_file = os.path.join("data", "processed", "normalized_comments.json")
    output_file = "normalized_comments_analyzed.json"
    
    print("Loading data...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Loaded {len(data)} entries.")

    # Use a standard Persian Sentiment Analysis model
    # HooshvareLab/bert-fa-base-uncased-sentiment-digikala
    model_name = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
    
    print(f"Loading BERT model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    try:
        # Create pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            tokenizer=model_name,
            device=device,
            truncation=True, 
            max_length=512
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Analyzing sentiment...")
   
    count = 0
    for entry in data:
        # The main comment is in 'description'
        message = entry.get("description", "")
        
        # Skip empty messages
        if not message or not isinstance(message, str) or len(message.strip()) < 2:
            entry["bert_sentiment"] = None
            entry["bert_score"] = None
            continue
            
        try:
           
            result = sentiment_pipeline(message)[0]
            label = result['label']
            score = float(result['score'])
            sentiment_map = {
                'recommended': 'POSITIVE',
                'not_recommended': 'NEGATIVE',
                'no_idea': 'NEUTRAL',
                'POSITIVE': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE',
                'LABEL_0': 'NEGATIVE', 
                'LABEL_1': 'POSITIVE'
            }
            
           
            mapped_label = sentiment_map.get(label, label)
            entry["bert_sentiment"] = mapped_label
            entry["bert_score"] = score
            
        except Exception as e:
            entry["bert_sentiment"] = "ERROR"
            entry["bert_score"] = 0.0
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(data)}")

    print("Saving results...")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Done! Updated {input_file} with sentiment data.")

if __name__ == "__main__":
    main()
