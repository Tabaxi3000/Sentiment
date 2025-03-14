import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import time
import os
import joblib
import torch
from tqdm import tqdm

def load_data(data_path, test_size=0.2):
    """Load and split data into train and test sets"""
    print(f"Loading data from {data_path}...")
    
    # Check file extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        X = df['text']
        y = df['score']
    elif data_path.endswith('.txt'):
        # For text files like amazon_cells_labelled.txt
        df = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'score'])
        X = df['text']
        y = df['score']
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return train_test_split(X, y, test_size=test_size, random_state=42)

def tune_logistic_regression(X_train, X_test, y_train, y_test):
    """Tune hyperparameters for Logistic Regression"""
    print("\nTuning Logistic Regression hyperparameters...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'tfidf__max_features': [3000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'saga'],
        'clf__max_iter': [1000]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    start_time = time.time()
    print("Starting grid search for Logistic Regression...")
    
    # Create a progress bar for visual feedback
    total_combinations = len(param_grid['tfidf__max_features']) * \
                         len(param_grid['tfidf__ngram_range']) * \
                         len(param_grid['clf__C']) * \
                         len(param_grid['clf__solver']) * 5  # 5 for CV folds
    
    print(f"Total parameter combinations to try: {total_combinations}")
    
    # Fit without monkey patching
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    
    print("Evaluating on test set...")
    y_pred = []
    for text in tqdm(X_test, desc="Predicting"):
        pred = best_model.predict([text])[0]
        y_pred.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    
    return {
        'model_name': 'logistic_regression',
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'tuning_time': tuning_time,
        'best_model': best_model
    }

def tune_adaboost(X_train, X_test, y_train, y_test):
    """Tune hyperparameters for AdaBoost"""
    print("\nTuning AdaBoost hyperparameters...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', AdaBoostClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 1.0]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    start_time = time.time()
    print("Starting grid search for AdaBoost...")
    
    # Create a progress bar for visual feedback
    total_combinations = len(param_grid['tfidf__max_features']) * \
                         len(param_grid['tfidf__ngram_range']) * \
                         len(param_grid['clf__n_estimators']) * \
                         len(param_grid['clf__learning_rate']) * 5  # 5 for CV folds
    
    print(f"Total parameter combinations to try: {total_combinations}")
    
    # Fit without monkey patching
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    
    print("Evaluating on test set...")
    y_pred = []
    for text in tqdm(X_test, desc="Predicting"):
        pred = best_model.predict([text])[0]
        y_pred.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    
    return {
        'model_name': 'adaboost',
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'tuning_time': tuning_time,
        'best_model': best_model
    }

def tune_svm(X_train, X_test, y_train, y_test):
    """Tune hyperparameters for SVM"""
    print("\nTuning SVM hyperparameters...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 'auto']
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    start_time = time.time()
    print("Starting grid search for SVM...")
    
    # Create a progress bar for visual feedback
    total_combinations = len(param_grid['tfidf__max_features']) * \
                         len(param_grid['tfidf__ngram_range']) * \
                         len(param_grid['clf__C']) * \
                         len(param_grid['clf__kernel']) * \
                         len(param_grid['clf__gamma']) * 5  # 5 for CV folds
    
    print(f"Total parameter combinations to try: {total_combinations}")
    
    # Fit without monkey patching
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    
    print("Evaluating on test set...")
    y_pred = []
    for text in tqdm(X_test, desc="Predicting"):
        pred = best_model.predict([text])[0]
        y_pred.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    
    return {
        'model_name': 'svm',
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'tuning_time': tuning_time,
        'best_model': best_model
    }

def save_tuned_model(result, dataset_name):
    """Save tuned model and results"""
    # Create directories
    os.makedirs('models/tuned', exist_ok=True)
    os.makedirs('results/tuning', exist_ok=True)
    
    # Save model
    model_path = f"models/tuned/{result['model_name']}_{dataset_name}_tuned.joblib"
    joblib.dump(result['best_model'], model_path)
    print(f"\nTuned model saved to {model_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': [result['model_name']],
        'Best CV Score': [result['best_cv_score']],
        'Test Accuracy': [result['test_accuracy']],
        'Tuning Time (s)': [result['tuning_time']]
    })
    
    # Add best parameters as columns
    for param, value in result['best_params'].items():
        results_df[f"Best {param}"] = [value]
    
    # Save to CSV
    results_path = f"results/tuning/{result['model_name']}_{dataset_name}_tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Tuning results saved to {results_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - Tuned {result['model_name']} on {dataset_name}")
    
    # Save plot
    cm_path = f"results/tuning/{result['model_name']}_{dataset_name}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

def compare_tuning_results(results, dataset_name):
    """Compare tuning results across models"""
    # Create comparison dataframe
    comparison = {
        'Model': [],
        'Best CV Score': [],
        'Test Accuracy': [],
        'Tuning Time (s)': []
    }
    
    for result in results:
        comparison['Model'].append(result['model_name'])
        comparison['Best CV Score'].append(result['best_cv_score'])
        comparison['Test Accuracy'].append(result['test_accuracy'])
        comparison['Tuning Time (s)'].append(result['tuning_time'])
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison)
    
    # Sort by test accuracy
    df = df.sort_values('Test Accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(f"results/tuning/{dataset_name}_tuning_comparison.csv", index=False)
    print(f"Comparison results saved to results/tuning/{dataset_name}_tuning_comparison.csv")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 1, 1)
    
    # Create bar plot
    bars = plt.bar(df['Model'], df['Test Accuracy'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom')
    
    plt.title(f'Model Accuracy Comparison on {dataset_name} Dataset (Tuned)')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Training time comparison
    plt.subplot(2, 1, 2)
    
    # Create bar plot
    bars = plt.bar(df['Model'], df['Tuning Time (s)'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.2f}", ha='center', va='bottom')
    
    plt.title(f'Model Tuning Time Comparison on {dataset_name} Dataset')
    plt.ylabel('Tuning Time (s)')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"results/tuning/{dataset_name}_tuning_comparison.png")
    plt.close()
    print(f"Comparison plot saved to results/tuning/{dataset_name}_tuning_comparison.png")
    
    print(f"\nModel tuning comparison for {dataset_name} dataset:")
    print(df)
    
    return df

def run_hyperparameter_tuning(dataset_path, dataset_name):
    """Run hyperparameter tuning for all models on a dataset"""
    # Load and split data
    X_train, X_test, y_train, y_test = load_data(dataset_path)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"{'='*50}\n")
    
    # Tune Logistic Regression
    logistic_result = tune_logistic_regression(X_train, X_test, y_train, y_test)
    save_tuned_model(logistic_result, dataset_name)
    
    # Tune AdaBoost
    adaboost_result = tune_adaboost(X_train, X_test, y_train, y_test)
    save_tuned_model(adaboost_result, dataset_name)
    
    # Tune SVM
    svm_result = tune_svm(X_train, X_test, y_train, y_test)
    save_tuned_model(svm_result, dataset_name)
    
    # Compare results
    results = [logistic_result, adaboost_result, svm_result]
    comparison_df = compare_tuning_results(results, dataset_name)
    
    return results

if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set CUDA options for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("GPU not available, using CPU")
    
    # Define datasets
    datasets = {
        'airlines': 'data/airlines_data.csv',
        'amazon': 'data/amazon_cells_labelled.txt',
        'yelp': 'data/yelp_labelled.txt'
    }
    
    # Track overall progress
    total_datasets = len(datasets)
    completed_datasets = 0
    
    # Run hyperparameter tuning for each dataset
    dataset_progress = tqdm(datasets.items(), desc="Datasets", position=0, leave=True)
    for dataset_name, dataset_path in dataset_progress:
        dataset_progress.set_description(f"Dataset: {dataset_name} ({completed_datasets+1}/{total_datasets})")
        
        print(f"\nRunning hyperparameter tuning for {dataset_name} dataset...")
        results = run_hyperparameter_tuning(dataset_path, dataset_name)
        
        completed_datasets += 1
        dataset_progress.set_postfix({'completed': f"{completed_datasets}/{total_datasets}"})
    
    print("\nHyperparameter tuning completed for all datasets.")
    print("All results are saved in the 'results/tuning' directory.") 