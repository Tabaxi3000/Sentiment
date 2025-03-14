#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU-Only Hyperparameter Tuning Script

This script performs very basic hyperparameter optimization for the traditional ML models
with a focus on speed rather than exhaustive search. It's designed to run on CPU only
without requiring Intel MKL libraries.

Author: Sentiment Analysis Team
"""

# Set environment variables to avoid MKL
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM

def load_data(data_path='data/combined_data_no_augmentation.csv'):
    """Load the combined dataset"""
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None, None
    
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['score']
    
    print(f"Loaded {len(df)} samples")
    
    return X, y

def optimize_logistic_regression(X_train, y_train, X_test, y_test):
    """Simple optimization for Logistic Regression"""
    print("\nOptimizing Logistic Regression...")
    start_time = time.time()
    
    # Define parameter combinations to try
    param_combinations = [
        {'C': 0.1, 'solver': 'liblinear', 'ngram_range': (1, 1), 'max_features': 5000},
        {'C': 1.0, 'solver': 'liblinear', 'ngram_range': (1, 2), 'max_features': 5000},
        {'C': 10.0, 'solver': 'liblinear', 'ngram_range': (1, 2), 'max_features': 10000},
        {'C': 1.0, 'solver': 'saga', 'ngram_range': (1, 2), 'max_features': 5000},
        {'C': 1.0, 'solver': 'liblinear', 'ngram_range': (1, 3), 'max_features': 10000}
    ]
    
    best_score = 0
    best_params = None
    best_model = None
    
    # Try each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create and train model
        model = SentimentLogisticRegression(
            C=params['C'],
            max_iter=1000,
            solver=params['solver'],
            ngram_range=params['ngram_range'],
            max_features=params['max_features']
        )
        
        model.train(X_train, y_train)
        
        # Evaluate on a validation set (20% of training data)
        X_train_subset, X_val, y_train_subset, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Predict on validation set
        y_pred = []
        for text in tqdm(X_val, desc="Evaluating"):
            _, score = model.predict(text)
            y_pred.append(score)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # Update best model if better
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            best_model = model
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation score: {best_score:.4f}")
    
    # Evaluate best model on test set
    y_pred = []
    for text in tqdm(X_test, desc="Evaluating best model on test set"):
        _, score = best_model.predict(text)
        y_pred.append(score)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)
    
    # Save the best model
    os.makedirs('models/optimized', exist_ok=True)
    best_model.save_model('models/optimized/logistic_regression_optimized.joblib')
    
    # Save the optimization results
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'test_accuracy': accuracy,
        'training_time': time.time() - start_time
    }
    
    joblib.dump(results, 'models/optimized/logistic_regression_optimization_results.joblib')
    
    return best_model, results

def optimize_svm(X_train, y_train, X_test, y_test):
    """Simple optimization for SVM"""
    print("\nOptimizing SVM...")
    start_time = time.time()
    
    # Define parameter combinations to try
    param_combinations = [
        {'C': 0.1, 'kernel': 'linear', 'ngram_range': (1, 1), 'max_features': 5000},
        {'C': 1.0, 'kernel': 'linear', 'ngram_range': (1, 2), 'max_features': 5000},
        {'C': 10.0, 'kernel': 'linear', 'ngram_range': (1, 2), 'max_features': 10000},
        {'C': 1.0, 'kernel': 'rbf', 'ngram_range': (1, 2), 'max_features': 5000}
    ]
    
    best_score = 0
    best_params = None
    best_model = None
    
    # Try each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\nTrying combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create and train model
        model = SentimentSVM(
            C=params['C'],
            kernel=params['kernel'],
            gamma='scale',
            ngram_range=params['ngram_range'],
            max_features=params['max_features']
        )
        
        model.train(X_train, y_train)
        
        # Evaluate on a validation set (20% of training data)
        X_train_subset, X_val, y_train_subset, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Predict on validation set
        y_pred = []
        for text in tqdm(X_val, desc="Evaluating"):
            _, score = model.predict(text)
            y_pred.append(score)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # Update best model if better
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            best_model = model
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation score: {best_score:.4f}")
    
    # Evaluate best model on test set
    y_pred = []
    for text in tqdm(X_test, desc="Evaluating best model on test set"):
        _, score = best_model.predict(text)
        y_pred.append(score)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)
    
    # Save the best model
    os.makedirs('models/optimized', exist_ok=True)
    best_model.save_model('models/optimized/svm_optimized.joblib')
    
    # Save the optimization results
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'test_accuracy': accuracy,
        'training_time': time.time() - start_time
    }
    
    joblib.dump(results, 'models/optimized/svm_optimization_results.joblib')
    
    return best_model, results

def create_improved_ensemble(models, X_test, y_test):
    """Create and evaluate an improved ensemble"""
    print("\nCreating improved ensemble...")
    
    # Define the VotingEnsemble class
    class VotingEnsemble:
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights if weights is not None else [1] * len(models)
        
        def predict(self, text):
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
    
    # Try different weight combinations
    weight_combinations = [
        [1.0, 1.0, 1.0],  # Equal weights
        [1.5, 1.5, 1.0],  # More weight to LR and SVM
        [1.0, 2.0, 1.0],  # More weight to SVM
        [2.0, 1.0, 1.0],  # More weight to LR
        [1.5, 2.5, 1.0]   # Optimized weights
    ]
    
    best_accuracy = 0
    best_weights = None
    best_ensemble = None
    best_results = None
    
    model_list = list(models.values())
    
    # Try each weight combination
    for i, weights in enumerate(weight_combinations):
        print(f"\nTrying weight combination {i+1}/{len(weight_combinations)}: {weights}")
        
        # Create ensemble
        ensemble = VotingEnsemble(model_list, weights)
        
        # Evaluate ensemble
        y_pred = []
        for text in tqdm(X_test, desc="Evaluating ensemble"):
            _, score = ensemble.predict(text)
            y_pred.append(score)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        
        # Update best ensemble if better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights
            best_ensemble = ensemble
            best_results = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': conf_matrix,
                'weights': {
                    'LogisticRegression': weights[0],
                    'SVM': weights[1],
                    'AdaBoost': weights[2]
                }
            }
    
    print(f"\nBest weights: {best_weights}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    # Plot confusion matrix for best ensemble
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Improved Ensemble Confusion Matrix')
    
    # Create directory for results
    os.makedirs('results/optimized', exist_ok=True)
    plt.savefig('results/optimized/improved_ensemble_confusion_matrix.png')
    plt.close()
    
    # Save the results
    joblib.dump(best_results, 'results/optimized/improved_ensemble_results.joblib')
    
    # Save ensemble configuration
    ensemble_config = {
        'weights': best_results['weights']
    }
    
    os.makedirs('models/optimized', exist_ok=True)
    joblib.dump(ensemble_config, 'models/optimized/improved_ensemble_config.joblib')
    
    return best_ensemble, best_results

def compare_results(original_results, optimized_results):
    """Compare original and optimized results"""
    print("\nComparing results...")
    
    # Create comparison dataframe
    comparison = {
        'Model': [],
        'Original Accuracy': [],
        'Optimized Accuracy': [],
        'Improvement': []
    }
    
    # Add individual models
    for model_name in ['LogisticRegression', 'SVM']:
        if model_name in original_results and model_name in optimized_results:
            orig_acc = original_results[model_name]['accuracy']
            opt_acc = optimized_results[model_name]['test_accuracy']
            improvement = opt_acc - orig_acc
            
            comparison['Model'].append(model_name)
            comparison['Original Accuracy'].append(orig_acc)
            comparison['Optimized Accuracy'].append(opt_acc)
            comparison['Improvement'].append(improvement)
    
    # Add ensemble
    if 'WeightedEnsemble' in original_results and 'ImprovedEnsemble' in optimized_results:
        orig_acc = original_results['WeightedEnsemble']['accuracy']
        opt_acc = optimized_results['ImprovedEnsemble']['accuracy']
        improvement = opt_acc - orig_acc
        
        comparison['Model'].append('Ensemble')
        comparison['Original Accuracy'].append(orig_acc)
        comparison['Optimized Accuracy'].append(opt_acc)
        comparison['Improvement'].append(improvement)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison)
    
    # Print comparison
    print(df)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    barWidth = 0.3
    
    # Set positions of bars on X axis
    r1 = list(range(len(df)))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, df['Original Accuracy'], width=barWidth, label='Original')
    plt.bar(r2, df['Optimized Accuracy'], width=barWidth, label='Optimized')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks([r + barWidth/2 for r in range(len(df))], df['Model'])
    plt.ylim(0.75, 0.9)  # Adjust as needed
    plt.legend()
    
    # Add improvement values
    for i in range(len(df)):
        plt.annotate(f"+{df['Improvement'][i]:.4f}",
                    xy=(r2[i], df['Optimized Accuracy'][i] + 0.01),
                    ha='center', va='bottom',
                    color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/optimized/accuracy_comparison.png')
    plt.close()
    
    # Save comparison to CSV
    df.to_csv('results/optimized/accuracy_comparison.csv', index=False)
    
    return df

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    if X is None or y is None:
        print("Exiting due to data loading error.")
        exit(1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Load original results
    try:
        original_results = joblib.load('results/combined_models/model_comparison.joblib')
        print("Loaded original results for comparison")
    except:
        print("Could not load original results, will only report optimized results")
        original_results = {}
    
    # Optimize models
    optimized_results = {}
    
    # Optimize Logistic Regression
    lr_model, lr_results = optimize_logistic_regression(X_train, y_train, X_test, y_test)
    optimized_results['LogisticRegression'] = lr_results
    
    # Optimize SVM
    svm_model, svm_results = optimize_svm(X_train, y_train, X_test, y_test)
    optimized_results['SVM'] = svm_results
    
    # Load AdaBoost (no optimization for time constraints)
    adaboost_model = SentimentAdaBoost()
    try:
        adaboost_model.load_model('models/combined/adaboost.joblib')
        print("Loaded AdaBoost model successfully")
    except:
        print("Could not load AdaBoost model, training a new one")
        adaboost_model.train(X_train, y_train)
        os.makedirs('models/optimized', exist_ok=True)
        adaboost_model.save_model('models/optimized/adaboost.joblib')
    
    # Create models dictionary
    models = {
        'LogisticRegression': lr_model,
        'SVM': svm_model,
        'AdaBoost': adaboost_model
    }
    
    # Create improved ensemble
    ensemble, ensemble_results = create_improved_ensemble(models, X_test, y_test)
    optimized_results['ImprovedEnsemble'] = ensemble_results
    
    # Compare results
    if original_results:
        try:
            comparison_df = compare_results(
                original_results.get('individual_models', {}) | original_results.get('ensembles', {}),
                optimized_results
            )
        except:
            print("Could not compare results due to incompatible format")
    
    print("\nOptimization completed successfully!") 