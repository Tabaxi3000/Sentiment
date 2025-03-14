#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Prediction Script for Combined Models

This script demonstrates how to use the trained models from the combined dataset
to predict sentiment for new text inputs.

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

class VotingEnsemble:
    """Voting Ensemble for sentiment analysis models"""
    
    def __init__(self, models, weights=None):
        """
        Initialize the voting ensemble
        
        Args:
            models: List of sentiment analysis models
            weights: List of weights for each model (optional)
        """
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)
        
    def predict(self, text):
        """
        Predict sentiment using weighted voting
        
        Args:
            text: Input text
            
        Returns:
            sentiment: "positive" or "negative"
            score: 1 for positive, 0 for negative
        """
        votes = []
        
        for i, model in enumerate(self.models):
            sentiment, score = model.predict(text)
            votes.append((score, self.weights[i]))
        
        # Calculate weighted vote
        weighted_sum = sum(vote * weight for vote, weight in votes)
        total_weight = sum(self.weights)
        
        # Determine final prediction
        if weighted_sum / total_weight >= 0.5:
            return "positive", 1
        else:
            return "negative", 0

def load_model(model_type, model_dir='models/combined'):
    """
    Load a trained model
    
    Args:
        model_type: Type of model to load ('lr', 'svm', 'adaboost', 'ensemble')
        model_dir: Directory containing the trained models
        
    Returns:
        Loaded model
    """
    model_paths = {
        'lr': os.path.join(model_dir, 'logisticregression.joblib'),
        'svm': os.path.join(model_dir, 'svm.joblib'),
        'adaboost': os.path.join(model_dir, 'adaboost.joblib')
    }
    
    if model_type not in model_paths and model_type != 'ensemble' and model_type != 'weighted':
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available models: {', '.join(list(model_paths.keys()) + ['ensemble', 'weighted'])}")
        return None
    
    try:
        if model_type == 'ensemble' or model_type == 'weighted':
            # Load all models for ensemble
            models = []
            
            # Load Logistic Regression
            if os.path.exists(model_paths['lr']):
                lr_model = SentimentLogisticRegression()
                lr_model.load_model(model_paths['lr'])
                models.append(lr_model)
            
            # Load SVM
            if os.path.exists(model_paths['svm']):
                svm_model = SentimentSVM()
                svm_model.load_model(model_paths['svm'])
                models.append(svm_model)
            
            # Load AdaBoost
            if os.path.exists(model_paths['adaboost']):
                adaboost_model = SentimentAdaBoost()
                adaboost_model.load_model(model_paths['adaboost'])
                models.append(adaboost_model)
            
            if not models:
                print("Error: No models found for ensemble")
                return None
            
            # Create ensemble
            if model_type == 'weighted':
                weights = [1, 2, 1]  # More weight to SVM
                return VotingEnsemble(models, weights)
            else:
                return VotingEnsemble(models)
        else:
            # Load individual model
            if not os.path.exists(model_paths[model_type]):
                print(f"Error: Model file '{model_paths[model_type]}' not found")
                return None
            
            if model_type == 'lr':
                model = SentimentLogisticRegression()
            elif model_type == 'svm':
                model = SentimentSVM()
            elif model_type == 'adaboost':
                model = SentimentAdaBoost()
            
            model.load_model(model_paths[model_type])
            return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_sentiment(text, model):
    """
    Predict sentiment for a given text
    
    Args:
        text: Input text
        model: Trained model
        
    Returns:
        sentiment: "positive" or "negative"
        score: 1 for positive, 0 for negative
    """
    try:
        sentiment, score = model.predict(text)
        return sentiment, score
    except Exception as e:
        print(f"Error predicting sentiment: {str(e)}")
        return "error", -1

def predict_from_file(input_file, output_file, model):
    """
    Predict sentiment for texts in a file
    
    Args:
        input_file: Path to input file (one text per line)
        output_file: Path to output file
        model: Trained model
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Predict sentiment for each text
        results = []
        for text in tqdm(texts, desc="Predicting sentiment"):
            sentiment, score = predict_sentiment(text, model)
            results.append((text, sentiment, score))
        
        # Write results to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Text\tSentiment\tScore\n")
            for text, sentiment, score in results:
                f.write(f"{text}\t{sentiment}\t{score}\n")
        
        print(f"Predictions saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def interactive_mode(model):
    """
    Interactive mode for sentiment prediction
    
    Args:
        model: Trained model
    """
    print("\nSentiment Analysis Interactive Mode")
    print("Enter text to predict sentiment (type 'exit' to quit)")
    
    while True:
        try:
            text = input("\nEnter text: ")
            
            if text.lower() == 'exit':
                break
            
            if not text.strip():
                continue
            
            sentiment, score = predict_sentiment(text, model)
            print(f"Sentiment: {sentiment} (Score: {score})")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sentiment Analysis Prediction")
    parser.add_argument("--model", type=str, default="weighted", 
                        choices=["lr", "svm", "adaboost", "ensemble", "weighted"],
                        help="Model to use for prediction")
    parser.add_argument("--input", type=str, help="Input file (one text per line)")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--text", type=str, help="Single text to predict")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if model is None:
        return
    
    # Predict sentiment
    if args.input and args.output:
        predict_from_file(args.input, args.output, model)
    elif args.text:
        sentiment, score = predict_sentiment(args.text, model)
        print(f"Text: {args.text}")
        print(f"Sentiment: {sentiment} (Score: {score})")
    elif args.interactive:
        interactive_mode(model)
    else:
        interactive_mode(model)

if __name__ == "__main__":
    main() 