import json
import pandas as pd
import os
from pathlib import Path

def process_evaluation_dataset(json_file_path):
    """
    Process the evaluation dataset (news/blogs/github) into CSV format
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_items = []
    
    for item in data['items']:
        processed_items.append({
            'markdown': item['markdown'],
            'language': item['language'],
            'type': item['type'],  # news/github/blogs
            'url': item.get('url', ''),
            'source': 'evaluation',
            'domain': item.get('domain', ''),
            'title': item.get('title', ''),
            'word_count': item.get('word_count', 0),
            'content_length': item.get('content_length', 0)
        })
    
    return pd.DataFrame(processed_items)

def process_wikipedia_dataset(dataset_folder):
    """
    Process the Wikipedia dataset into CSV format
    """
    processed_items = []
    
    # Load index to get file list
    with open(os.path.join(dataset_folder, 'index.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Process each language file
    for filename in index['metadata']['files'].keys():
        if filename.startswith('wikipedia_') and filename.endswith('.json'):
            filepath = os.path.join(dataset_folder, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                lang_data = json.load(f)
            
            for article in lang_data['articles']:
                processed_items.append({
                    'markdown': article['markdown'],
                    'language': article['language'],
                    'url': article.get('url', ''),
                    'type': 'wiki',
                    'source': 'wikipedia',
                    'domain': 'wikipedia.org',
                    'title': article.get('topic', ''),
                    'word_count': article.get('word_count', 0),
                    'content_length': article.get('content_length', 0),
                    'category': article.get('category', '')
                })
    
    return pd.DataFrame(processed_items)

def create_train_validation_split(df, train_ratio=0.8, random_state=42):
    """
    Split Wikipedia data into train and validation sets
    """
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    split_point = int(len(df_shuffled) * train_ratio)
    
    train_df = df_shuffled[:split_point].copy()
    val_df = df_shuffled[split_point:].copy()
    
    # Add split identifier
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    
    return train_df, val_df

def main():
    # File paths
    evaluation_json = r'dataset/raw/evaluation_dataset/evaluation_dataset.json'  # Update path as needed
    wikipedia_folder = r'dataset/raw/wikipedia_dataset'        # Update path as needed

    print("Processing evaluation dataset...")
    eval_df = process_evaluation_dataset(evaluation_json)
    print(f"Evaluation dataset: {len(eval_df)} items")
    
    print("Processing Wikipedia dataset...")
    wiki_df = process_wikipedia_dataset(wikipedia_folder)
    print(f"Wikipedia dataset: {len(wiki_df)} items")
    
    print("Creating train/validation split...")
    train_df, val_df = create_train_validation_split(wiki_df, train_ratio=0.8)
    print(f"Train: {len(train_df)} items, Validation: {len(val_df)} items")
    
    # Combine train and validation
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Save to CSV files
    print("Saving CSV files...")
    train_val_df.to_csv('train_validation_dataset.csv', index=False, encoding='utf-8')
    eval_df.to_csv('evaluation_dataset.csv', index=False, encoding='utf-8')
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Train+Validation CSV: {len(train_val_df)} items")
    print("  Languages:", train_val_df['language'].value_counts().to_dict())
    print("  Types:", train_val_df['type'].value_counts().to_dict())
    print("  Split:", train_val_df['split'].value_counts().to_dict())
    
    print(f"\nEvaluation CSV: {len(eval_df)} items")
    print("  Languages:", eval_df['language'].value_counts().to_dict())
    print("  Types:", eval_df['type'].value_counts().to_dict())
    
    print(f"\nFiles saved:")
    print(f"  - train_validation_dataset.csv")
    print(f"  - evaluation_dataset.csv")

if __name__ == "__main__":
    main()