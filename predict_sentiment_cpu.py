#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Prediction Script (CPU Version)

This script demonstrates how to use the optimized models for sentiment prediction.
It loads the best models (optimized or ensemble) and allows the user to predict
sentiment for input text. This version only uses CPU-compatible models.

Author: Sentiment Analysis Team
"""

import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM

# Import ensemble classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_models_cpu import VotingEnsemble, StackingEnsemble

def load_model(model_type, model_path=None):
    """
    Load a model for sentiment prediction
    
    Args:
        model_type (str): Type of model to load ('lr', 'svm', 'adaboost', 'voting', 'stacking')
        model_path (str, optional): Path to the model file. If None, uses the default path.
        
    Returns:
        object: Loaded model
    """
    # Define default paths
    default_paths = {
        'lr': 'models/optimized/logistic_regression_optimized.joblib',
        'svm': 'models/optimized/svm_optimized.joblib',
        'adaboost': 'models/optimized/adaboost_optimized.joblib',
        'voting': 'models/ensemble/voting_ensemble_config.joblib',
        'stacking': 'models/ensemble/stacking_ensemble_config.joblib'
    }
    
    # Use default path if not provided
    if model_path is None:
        model_path = default_paths.get(model_type)
        if model_path is None:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Load model based on type
    if model_type == 'lr':
        model = SentimentLogisticRegression()
        model.load_model(model_path)
    elif model_type == 'svm':
        model = SentimentSVM()
        model.load_model(model_path)
    elif model_type == 'adaboost':
        model = SentimentAdaBoost()
        model.load_model(model_path)
    elif model_type == 'voting':
        # Load voting ensemble configuration
        config = joblib.load(model_path)
        
        # Load base models
        models = {}
        for name, path in config['model_paths'].items():
            if 'logistic_regression' in name:
                models[name] = SentimentLogisticRegression()
                models[name].load_model(path)
            elif 'svm' in name:
                models[name] = SentimentSVM()
                models[name].load_model(path)
            elif 'adaboost' in name:
                models[name] = SentimentAdaBoost()
                models[name].load_model(path)
        
        # Create voting ensemble
        model_list = list(models.values())
        weights = [config['weights'].get(name, 1) for name in models.keys()]
        model = VotingEnsemble(model_list, weights)
    elif model_type == 'stacking':
        # Load stacking ensemble configuration
        config = joblib.load(model_path)
        
        # Load base models
        base_models = []
        for name, path in config['base_model_paths'].items():
            if 'logistic_regression' in name:
                base_model = SentimentLogisticRegression()
                base_model.load_model(path)
                base_models.append(base_model)
            elif 'svm' in name:
                base_model = SentimentSVM()
                base_model.load_model(path)
                base_models.append(base_model)
            elif 'adaboost' in name:
                base_model = SentimentAdaBoost()
                base_model.load_model(path)
                base_models.append(base_model)
        
        # Create meta-model
        meta_model_type = config['meta_model_type']
        if meta_model_type == 'logistic_regression':
            meta_model = SentimentLogisticRegression()
        elif meta_model_type == 'svm':
            meta_model = SentimentSVM()
        elif meta_model_type == 'adaboost':
            meta_model = SentimentAdaBoost()
        else:
            raise ValueError(f"Unsupported meta-model type: {meta_model_type}")
        
        # Create stacking ensemble
        model = StackingEnsemble(base_models, meta_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model loaded successfully.")
    return model

def predict_sentiment(model, text):
    """
    Predict sentiment for a single text
    
    Args:
        model: Sentiment model
        text (str): Input text
        
    Returns:
        tuple: (sentiment, score)
    """
    sentiment, score = model.predict(text)
    return sentiment, score

def predict_from_file(model, input_file, output_file=None):
    """
    Predict sentiment for texts in a file
    
    Args:
        model: Sentiment model
        input_file (str): Path to input file (one text per line)
        output_file (str, optional): Path to output file. If None, prints to console.
        
    Returns:
        pandas.DataFrame: Predictions
    """
    # Read input file
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")
        texts = df['text'].tolist()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    print(f"Predicting sentiment for {len(texts)} texts...")
    
    # Predict sentiment
    results = []
    for text in tqdm(texts, desc="Predicting"):
        sentiment, score = predict_sentiment(model, text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'score': score
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save or print results
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("\nResults:")
        for i, row in results_df.iterrows():
            print(f"{i+1}. Text: {row['text']}")
            print(f"   Sentiment: {row['sentiment']} (Score: {row['score']:.4f})")
            print()
    
    return results_df

def interactive_mode(model):
    """
    Run interactive mode for sentiment prediction
    
    Args:
        model: Sentiment model
    """
    print("\nInteractive Sentiment Prediction Mode (CPU Version)")
    print("Enter text to predict sentiment (type 'exit' to quit)")
    
    while True:
        print("\n" + "-"*80)
        text = input("Enter text: ")
        
        if text.lower() in ['exit', 'quit', 'q']:
            break
        
        if not text.strip():
            continue
        
        sentiment, score = predict_sentiment(model, text)
        
        print(f"Sentiment: {sentiment}")
        print(f"Score: {score:.4f}")
        
        # Print confidence level
        confidence = score if sentiment == "positive" else 1 - score
        confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
        print(f"Confidence: {confidence_level} ({confidence:.4f})")
    
    print("\nExiting interactive mode.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Sentiment Prediction Script (CPU Version)")
    
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["lr", "svm", "adaboost", "voting", "stacking"],
        help="Type of model to use"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model file"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file (one text per line or CSV with 'text' column)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to output file (CSV format)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.model_path)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(model)
    elif args.input_file:
        predict_from_file(model, args.input_file, args.output_file)
    else:
        # Default to interactive mode if no input file provided
        interactive_mode(model)

if __name__ == "__main__":
    main() 