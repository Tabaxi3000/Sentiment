#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Dataset Training Script

This script combines all available datasets without augmentation,
then trains and evaluates traditional machine learning models
and ensemble methods on the combined data.

Author: Sentiment Analysis Team
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM

def load_and_combine_datasets():
    """Load and combine all available datasets without augmentation"""
    print("Loading and combining all datasets...")
    
    datasets = []
    
    # Load airlines data
    airlines_path = 'data/airlines_data.csv'
    if os.path.exists(airlines_path):
        print(f"Loading {airlines_path}...")
        airlines_df = pd.read_csv(airlines_path)
        # Ensure the dataframe has the expected columns
        if 'text' in airlines_df.columns and 'airline_sentiment' in airlines_df.columns:
            # Map sentiment to binary score (1 for positive, 0 for negative)
            airlines_df['score'] = airlines_df['airline_sentiment'].apply(
                lambda x: 1 if x.lower() == 'positive' else 0
            )
            # Select only the needed columns
            airlines_df = airlines_df[['text', 'score']]
            datasets.append(airlines_df)
            print(f"Added {len(airlines_df)} samples from airlines data")
    
    # Load Amazon reviews
    amazon_path = 'data/amazon_cells_labelled.txt'
    if os.path.exists(amazon_path):
        print(f"Loading {amazon_path}...")
        amazon_df = pd.read_csv(amazon_path, delimiter='\t', header=None, names=['text', 'score'])
        datasets.append(amazon_df)
        print(f"Added {len(amazon_df)} samples from Amazon reviews")
    
    # Load Yelp reviews
    yelp_path = 'data/yelp_labelled.txt'
    if os.path.exists(yelp_path):
        print(f"Loading {yelp_path}...")
        yelp_df = pd.read_csv(yelp_path, delimiter='\t', header=None, names=['text', 'score'])
        datasets.append(yelp_df)
        print(f"Added {len(yelp_df)} samples from Yelp reviews")
    
    # Load IMDB reviews
    imdb_path = 'data/imdb_labelled.txt'
    if os.path.exists(imdb_path):
        print(f"Loading {imdb_path}...")
        imdb_df = pd.read_csv(imdb_path, delimiter='\t', header=None, names=['text', 'score'])
        datasets.append(imdb_df)
        print(f"Added {len(imdb_df)} samples from IMDB reviews")
    
    # Check if we have any datasets
    if not datasets:
        print("Error: No datasets found!")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Remove any rows with NaN values
    combined_df = combined_df.dropna()
    
    # Save the combined dataset
    output_path = 'data/combined_data_no_augmentation.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined dataset size: {len(combined_df)} samples")
    print(f"Saved combined dataset to {output_path}")
    
    return combined_df

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    model.train(X_train, y_train)
    
    print(f"Evaluating {model_name}...")
    y_pred = []
    for text in tqdm(X_test, desc=f"Evaluating {model_name}"):
        _, score = model.predict(text)
        y_pred.append(score)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Create directory for results
    os.makedirs('results/combined_models', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'results/combined_models/{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Save the model
    os.makedirs('models/combined', exist_ok=True)
    model.save_model(f'models/combined/{model_name.lower()}.joblib')
    
    # Save the results
    results = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    joblib.dump(results, f'results/combined_models/{model_name.lower()}_results.joblib')
    
    return model, results

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

def create_voting_ensemble(models, weights=None):
    """Create a voting ensemble from the given models"""
    model_list = list(models.values())
    
    if weights is not None:
        weight_list = [weights.get(name, 1) for name in models.keys()]
    else:
        weight_list = None
    
    return VotingEnsemble(model_list, weight_list)

def evaluate_ensemble(ensemble, X_test, y_test, name="Ensemble"):
    """Evaluate the ensemble on the test set"""
    print(f"\nEvaluating {name}...")
    
    y_pred = []
    
    for text in tqdm(X_test, desc=f"Evaluating {name}"):
        sentiment, score = ensemble.predict(text)
        y_pred.append(score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'results/combined_models/{name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Save the results
    results = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    joblib.dump(results, f'results/combined_models/{name.lower()}_results.joblib')
    
    return results

def compare_models(models_results, ensembles_results):
    """Compare all models including the ensembles"""
    print("\nComparing all models...")
    
    # Extract accuracies
    accuracies = {name: results['accuracy'] for name, results in models_results.items()}
    
    # Add ensemble accuracies
    for name, results in ensembles_results.items():
        accuracies[name] = results['accuracy']
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/combined_models/model_comparison.png')
    plt.close()
    
    # Save comparison results
    comparison = {
        'individual_models': models_results,
        'ensembles': ensembles_results
    }
    
    joblib.dump(comparison, 'results/combined_models/model_comparison.joblib')
    
    print("Model comparison completed.")

if __name__ == "__main__":
    # Load and combine all datasets
    combined_df = load_and_combine_datasets()
    
    if combined_df is None:
        print("Exiting due to data loading error.")
        exit(1)
    
    # Prepare data
    X = combined_df['text']
    y = combined_df['score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train and evaluate models
    models = {}
    models_results = {}
    
    # Logistic Regression
    lr_model = SentimentLogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear',
        ngram_range=(1, 2),
        max_features=5000
    )
    lr_model, lr_results = train_and_evaluate_model(
        lr_model, "LogisticRegression", X_train, y_train, X_test, y_test
    )
    models['LogisticRegression'] = lr_model
    models_results['LogisticRegression'] = lr_results
    
    # SVM
    svm_model = SentimentSVM(
        C=1.0,
        kernel='linear',
        gamma='scale',
        ngram_range=(1, 2),
        max_features=5000
    )
    svm_model, svm_results = train_and_evaluate_model(
        svm_model, "SVM", X_train, y_train, X_test, y_test
    )
    models['SVM'] = svm_model
    models_results['SVM'] = svm_results
    
    # AdaBoost
    adaboost_model = SentimentAdaBoost(
        n_estimators=100,
        learning_rate=1.0,
        ngram_range=(1, 2),
        max_features=5000
    )
    adaboost_model, adaboost_results = train_and_evaluate_model(
        adaboost_model, "AdaBoost", X_train, y_train, X_test, y_test
    )
    models['AdaBoost'] = adaboost_model
    models_results['AdaBoost'] = adaboost_results
    
    # Create and evaluate ensembles
    ensembles_results = {}
    
    # Equal weights voting ensemble
    equal_weights_ensemble = create_voting_ensemble(models)
    equal_results = evaluate_ensemble(equal_weights_ensemble, X_test, y_test, "EqualWeightsEnsemble")
    ensembles_results["EqualWeightsEnsemble"] = equal_results
    
    # Custom weights voting ensemble
    custom_weights = {
        'LogisticRegression': 1,
        'SVM': 2,  # Give more weight to SVM as it typically performs better
        'AdaBoost': 1
    }
    
    weighted_ensemble = create_voting_ensemble(models, custom_weights)
    weighted_results = evaluate_ensemble(weighted_ensemble, X_test, y_test, "WeightedEnsemble")
    ensembles_results["WeightedEnsemble"] = weighted_results
    
    # Compare all models
    compare_models(models_results, ensembles_results)
    
    print("\nModel training and evaluation completed successfully!") 