#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Combination Script

This script combines multiple sentiment analysis datasets into a single dataset
for training and evaluation. It performs basic preprocessing but avoids
time-consuming augmentation techniques.

Author: Sentiment Analysis Team
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import time

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_dataset(file_path, dataset_name):
    """Load a dataset from a file path"""
    print(f"Loading {dataset_name} dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return None
    
    # Handle different file formats
    if file_path.endswith('.csv'):
        # For CSV files like airlines_data.csv
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        if 'text' in df.columns and 'score' in df.columns:
            # Already in the right format
            pass
        elif 'text' in df.columns and 'airline_sentiment' in df.columns:
            # Airlines dataset
            sentiment_map = {'positive': 1, 'negative': 0, 'neutral': None}
            df['score'] = df['airline_sentiment'].map(sentiment_map)
            # Remove neutral sentiments
            df = df[df['score'].notna()]
            df['score'] = df['score'].astype(int)
        else:
            print(f"Warning: {file_path} has an unsupported format. Skipping.")
            return None
    
    elif file_path.endswith('.txt'):
        # For text files like amazon_cells_labelled.txt
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'score'])
        except:
            print(f"Warning: {file_path} has an unsupported format. Skipping.")
            return None
    
    else:
        print(f"Warning: {file_path} has an unsupported extension. Skipping.")
        return None
    
    # Add dataset source column
    df['source'] = dataset_name
    
    # Keep only text and score columns
    df = df[['text', 'score', 'source']]
    
    print(f"Loaded {len(df)} samples from {dataset_name} dataset")
    return df

def basic_preprocess(text):
    """Perform basic preprocessing on text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(df):
    """Preprocess the entire dataset"""
    print("Preprocessing dataset...")
    start_time = time.time()
    
    # Apply basic preprocessing to text column
    df['text'] = df['text'].progress_apply(basic_preprocess)
    
    # Remove empty texts
    df = df[df['text'].str.strip().astype(bool)]
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Dataset size after preprocessing: {len(df)} samples")
    
    return df

def combine_datasets():
    """Combine all datasets into a single dataset"""
    # Define dataset paths and names
    datasets = [
        ('data/airlines_data.csv', 'airlines'),
        ('data/amazon_cells_labelled.txt', 'amazon'),
        ('data/yelp_labelled.txt', 'yelp'),
        ('data/imdb_labelled.txt', 'imdb')
    ]
    
    # Load all datasets
    dfs = []
    for file_path, dataset_name in datasets:
        df = load_dataset(file_path, dataset_name)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("Error: No datasets could be loaded.")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)} samples")
    
    # Enable tqdm for pandas
    tqdm.pandas(desc="Preprocessing")
    
    # Preprocess the combined dataset
    combined_df = preprocess_dataset(combined_df)
    
    # Shuffle the dataset
    combined_df = shuffle(combined_df, random_state=42)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(combined_df)}")
    
    # Count by source
    source_counts = combined_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} samples")
    
    # Count by sentiment
    sentiment_counts = combined_df['score'].value_counts()
    for sentiment, count in sentiment_counts.items():
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        print(f"  {sentiment_label}: {count} samples ({count/len(combined_df)*100:.1f}%)")
    
    return combined_df

def save_dataset(df, output_path):
    """Save the dataset to a CSV file"""
    print(f"Saving combined dataset to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path}")

if __name__ == "__main__":
    # Set output path
    output_path = 'data/combined_data_no_augmentation.csv'
    
    # Combine datasets
    combined_df = combine_datasets()
    
    if combined_df is not None:
        # Save the combined dataset
        save_dataset(combined_df, output_path)
        
        print("\nCombined dataset created successfully!")
    else:
        print("\nFailed to create combined dataset.") 