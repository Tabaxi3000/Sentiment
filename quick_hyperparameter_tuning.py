#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick Hyperparameter Tuning Script

This script performs rapid hyperparameter optimization for the traditional ML models
with a focus on the most impactful parameters. Designed to complete within 2-3 hours.

Author: Sentiment Analysis Team
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import time

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
    """Quick optimization for Logistic Regression"""
    print("\nOptimizing Logistic Regression...")
    start_time = time.time()
    
    # Create a wrapper for scikit-learn's GridSearchCV
    class LogisticRegressionWrapper:
        def __init__(self, C=1.0, max_iter=1000, solver='liblinear', ngram_range=(1, 2), max_features=5000):
            self.C = C
            self.max_iter = max_iter
            self.solver = solver
            self.ngram_range = ngram_range
            self.max_features = max_features
            self.model = None
        
        def fit(self, X, y):
            self.model = SentimentLogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver=self.solver,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
            self.model.train(X, y)
            return self
        
        def predict(self, X):
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            y_pred = []
            for text in X:
                _, score = self.model.predict(text)
                y_pred.append(score)
            
            return np.array(y_pred)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
            
        # Add get_params and set_params methods required by GridSearchCV
        def get_params(self, deep=True):
            return {
                'C': self.C,
                'max_iter': self.max_iter,
                'solver': self.solver,
                'ngram_range': self.ngram_range,
                'max_features': self.max_features
            }
        
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
    
    # Define parameter grid (limited for quick results)
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga'],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'max_features': [3000, 5000, 10000]
    }
    
    # Create base model
    base_model = LogisticRegressionWrapper()
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # Use 3-fold CV for speed
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Train model with best parameters
    best_model = SentimentLogisticRegression(
        C=best_params['C'],
        max_iter=1000,  # Fixed for stability
        solver=best_params['solver'],
        ngram_range=best_params['ngram_range'],
        max_features=best_params['max_features']
    )
    
    best_model.train(X_train, y_train)
    
    # Evaluate on test set
    y_pred = []
    for text in tqdm(X_test, desc="Evaluating best model"):
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
        'best_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'training_time': time.time() - start_time
    }
    
    joblib.dump(results, 'models/optimized/logistic_regression_optimization_results.joblib')
    
    return best_model, results

def optimize_svm(X_train, y_train, X_test, y_test):
    """Quick optimization for SVM"""
    print("\nOptimizing SVM...")
    start_time = time.time()
    
    # Create a wrapper for scikit-learn's GridSearchCV
    class SVMWrapper:
        def __init__(self, C=1.0, kernel='linear', gamma='scale', ngram_range=(1, 2), max_features=5000):
            self.C = C
            self.kernel = kernel
            self.gamma = gamma
            self.ngram_range = ngram_range
            self.max_features = max_features
            self.model = None
        
        def fit(self, X, y):
            self.model = SentimentSVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
            self.model.train(X, y)
            return self
        
        def predict(self, X):
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            y_pred = []
            for text in X:
                _, score = self.model.predict(text)
                y_pred.append(score)
            
            return np.array(y_pred)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
            
        # Add get_params and set_params methods required by GridSearchCV
        def get_params(self, deep=True):
            return {
                'C': self.C,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'ngram_range': self.ngram_range,
                'max_features': self.max_features
            }
        
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
    
    # Define parameter grid (limited for quick results)
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'max_features': [5000, 10000]
    }
    
    # Create base model
    base_model = SVMWrapper()
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # Use 3-fold CV for speed
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Train model with best parameters
    best_model = SentimentSVM(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma='scale',  # Fixed for stability
        ngram_range=best_params['ngram_range'],
        max_features=best_params['max_features']
    )
    
    best_model.train(X_train, y_train)
    
    # Evaluate on test set
    y_pred = []
    for text in tqdm(X_test, desc="Evaluating best model"):
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
        'best_score': grid_search.best_score_,
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
    
    # Create ensemble with optimized weights
    weights = {
        'LogisticRegression': 1.5,  # Increased weight
        'SVM': 2.5,                # Increased weight
        'AdaBoost': 1.0            # Same weight
    }
    
    model_list = list(models.values())
    weight_list = [weights.get(name, 1) for name in models.keys()]
    
    ensemble = VotingEnsemble(model_list, weight_list)
    
    # Evaluate ensemble
    y_pred = []
    for text in tqdm(X_test, desc="Evaluating ensemble"):
        _, score = ensemble.predict(text)
        y_pred.append(score)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
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
    results = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'weights': weights
    }
    
    joblib.dump(results, 'results/optimized/improved_ensemble_results.joblib')
    
    # Save ensemble configuration
    ensemble_config = {
        'weights': weights
    }
    
    os.makedirs('models/optimized', exist_ok=True)
    joblib.dump(ensemble_config, 'models/optimized/improved_ensemble_config.joblib')
    
    return ensemble, results

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
    r1 = np.arange(len(df))
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
    for i, p in enumerate(plt.gca().patches[len(df):]):
        plt.gca().annotate(f"+{df['Improvement'][i]:.4f}",
                          (p.get_x() + p.get_width()/2., p.get_height()),
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
    adaboost_model.load_model('models/combined/adaboost.joblib')
    
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
        comparison_df = compare_results(
            original_results.get('individual_models', {}) | original_results.get('ensembles', {}),
            optimized_results
        )
    
    print("\nOptimization completed successfully!") 