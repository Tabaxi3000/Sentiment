import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import joblib
import torch
from tqdm import tqdm

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM
from models.bert_sentiment import SentimentBERT
from models.lstm_sentiment import SentimentLSTM
from models.roberta_sentiment import SentimentRoBERTa

def load_data(data_path):
    """Load data from a CSV file"""
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
    
    return X, y

def initialize_model(model_name):
    """Initialize a model based on its name"""
    if model_name.lower() == 'bert':
        # Use pretrained BERT model for faster results
        # Using textattack/bert-base-uncased-SST-2 which is more reliable
        return SentimentBERT(epochs=2, use_pretrained=True, pretrained_model="textattack/bert-base-uncased-SST-2")
    elif model_name.lower() == 'lstm':
        return SentimentLSTM(epochs=3)
    elif model_name.lower() == 'roberta':
        # Use pretrained RoBERTa model for faster results
        return SentimentRoBERTa(epochs=2, use_pretrained=True, pretrained_model="cardiffnlp/twitter-roberta-base-sentiment")
    elif model_name.lower() == 'logistic':
        return SentimentLogisticRegression()
    elif model_name.lower() == 'adaboost':
        return SentimentAdaBoost()
    elif model_name.lower() == 'svm':
        return SentimentSVM()
    else:
        raise ValueError(f"Model {model_name} not implemented")

def cross_validate(model_name, X, y, n_splits=5):
    """
    Perform k-fold cross-validation
    
    Args:
        model_name (str): Name of the model to use
        X (pandas.Series): Input texts
        y (pandas.Series): Target labels
        n_splits (int): Number of folds for cross-validation
        
    Returns:
        dict: Dictionary with cross-validation results
    """
    print(f"\nPerforming {n_splits}-fold cross-validation for {model_name}...")
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics
    fold_accuracies = []
    fold_reports = []
    fold_cms = []
    fold_times = []
    
    # Create progress bar for folds
    fold_progress = tqdm(enumerate(kf.split(X)), total=n_splits, desc="Folds", position=0, leave=True)
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in fold_progress:
        fold_progress.set_description(f"Fold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize model
        model = initialize_model(model_name)
        
        # Train and evaluate
        start_time = time.time()
        
        # Train
        print(f"\nTraining {model_name} model...")
        model.train(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        report = model.evaluate(X_test, y_test)
        
        # Calculate accuracy
        y_pred = []
        
        # Use tqdm for prediction progress
        prediction_progress = tqdm(X_test, desc="Predicting", position=1, leave=False)
        for text in prediction_progress:
            sentiment, score = model.predict(text)
            y_pred.append(score)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        fold_accuracies.append(accuracy)
        fold_reports.append(report)
        fold_cms.append(cm)
        fold_times.append(training_time)
        
        # Update fold progress bar
        fold_progress.set_postfix({'accuracy': f"{accuracy:.4f}", 'time': f"{training_time:.2f}s"})
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available() and hasattr(model, 'model') and hasattr(model.model, 'to'):
            # For PyTorch models
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    # Calculate average metrics
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    avg_time = np.mean(fold_times)
    
    # Create average confusion matrix
    avg_cm = np.mean(fold_cms, axis=0)
    
    print(f"\nCross-validation results for {model_name}:")
    print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average training time: {avg_time:.2f} seconds")
    
    return {
        'model_name': model_name,
        'fold_accuracies': fold_accuracies,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'fold_reports': fold_reports,
        'fold_cms': fold_cms,
        'avg_cm': avg_cm,
        'fold_times': fold_times,
        'avg_time': avg_time
    }

def plot_cv_results(cv_results, dataset_name):
    """
    Plot cross-validation results
    
    Args:
        cv_results (dict): Dictionary with cross-validation results
        dataset_name (str): Name of the dataset
    """
    # Create directory for results
    os.makedirs('results/cross_validation', exist_ok=True)
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    
    # Create box plot
    plt.boxplot(cv_results['fold_accuracies'], labels=[cv_results['model_name']])
    
    # Add individual points
    plt.scatter([1] * len(cv_results['fold_accuracies']), cv_results['fold_accuracies'], 
                color='red', alpha=0.5)
    
    # Add mean line
    plt.axhline(cv_results['avg_accuracy'], color='blue', linestyle='--', 
                label=f"Mean: {cv_results['avg_accuracy']:.4f}")
    
    plt.title(f"Cross-validation Accuracy for {cv_results['model_name']} on {dataset_name}")
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig(f"results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_accuracy.png")
    plt.close()
    print(f"Accuracy plot saved to results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_accuracy.png")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cv_results['avg_cm'], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Average Confusion Matrix - {cv_results['model_name']} on {dataset_name}")
    
    # Save plot
    plt.savefig(f"results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_cm.png")
    plt.close()
    print(f"Confusion matrix saved to results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_cm.png")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Fold': list(range(1, len(cv_results['fold_accuracies']) + 1)),
        'Accuracy': cv_results['fold_accuracies'],
        'Training Time (s)': cv_results['fold_times']
    })
    
    results_df.loc['Mean'] = ['-', cv_results['avg_accuracy'], cv_results['avg_time']]
    results_df.loc['Std'] = ['-', cv_results['std_accuracy'], np.std(cv_results['fold_times'])]
    
    results_df.to_csv(f"results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_results.csv", index=False)
    print(f"Results saved to results/cross_validation/{cv_results['model_name']}_{dataset_name}_cv_results.csv")

def run_cross_validation(dataset_path, dataset_name, models_to_validate, n_splits=5):
    """
    Run cross-validation for multiple models on a dataset
    
    Args:
        dataset_path (str): Path to the dataset
        dataset_name (str): Name of the dataset
        models_to_validate (list): List of model names to validate
        n_splits (int): Number of folds for cross-validation
    """
    # Load data
    X, y = load_data(dataset_path)
    
    # Create results directory
    os.makedirs('results/cross_validation', exist_ok=True)
    
    # Store results
    all_cv_results = {}
    
    # Create progress bar for models
    model_progress = tqdm(models_to_validate, desc="Models", position=0, leave=True)
    
    # Run cross-validation for each model
    for i, model_name in enumerate(model_progress):
        model_progress.set_description(f"Model: {model_name} ({i+1}/{len(models_to_validate)})")
        
        cv_results = cross_validate(model_name, X, y, n_splits)
        all_cv_results[model_name] = cv_results
        
        # Plot results
        plot_cv_results(cv_results, dataset_name)
        
        # Update model progress bar
        model_progress.set_postfix({'accuracy': f"{cv_results['avg_accuracy']:.4f}"})
    
    # Compare models
    compare_cv_results(all_cv_results, dataset_name)
    
    return all_cv_results

def compare_cv_results(all_cv_results, dataset_name):
    """
    Compare cross-validation results across models
    
    Args:
        all_cv_results (dict): Dictionary with cross-validation results for multiple models
        dataset_name (str): Name of the dataset
    """
    # Create comparison dataframe
    comparison = {
        'Model': [],
        'Average Accuracy': [],
        'Accuracy Std': [],
        'Average Training Time (s)': []
    }
    
    for model_name, cv_results in all_cv_results.items():
        comparison['Model'].append(model_name)
        comparison['Average Accuracy'].append(cv_results['avg_accuracy'])
        comparison['Accuracy Std'].append(cv_results['std_accuracy'])
        comparison['Average Training Time (s)'].append(cv_results['avg_time'])
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison)
    
    # Sort by average accuracy
    df = df.sort_values('Average Accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(f"results/cross_validation/{dataset_name}_model_comparison.csv", index=False)
    print(f"Model comparison saved to results/cross_validation/{dataset_name}_model_comparison.csv")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 1, 1)
    
    # Create bar plot with error bars
    bars = plt.bar(df['Model'], df['Average Accuracy'], yerr=df['Accuracy Std'],
                  capsize=5, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha='center', va='bottom')
    
    plt.title(f'Model Accuracy Comparison on {dataset_name} Dataset')
    plt.ylabel('Average Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Training time comparison
    plt.subplot(2, 1, 2)
    
    # Create bar plot
    bars = plt.bar(df['Model'], df['Average Training Time (s)'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.2f}", ha='center', va='bottom')
    
    plt.title(f'Model Training Time Comparison on {dataset_name} Dataset')
    plt.ylabel('Average Training Time (s)')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"results/cross_validation/{dataset_name}_model_comparison.png")
    plt.close()
    print(f"Comparison plot saved to results/cross_validation/{dataset_name}_model_comparison.png")
    
    print(f"\nModel comparison for {dataset_name} dataset:")
    print(df)
    
    return df

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
    
    # Define models to validate
    # For traditional ML models (faster)
    traditional_models = ['logistic', 'adaboost', 'svm']
    
    # For deep learning models (more resource-intensive)
    deep_learning_models = ['lstm', 'bert', 'roberta']
    
    # Number of folds for cross-validation
    n_splits = 5
    
    # Track overall progress
    total_datasets = 3 + 1  # 3 regular datasets + 1 subset
    completed_datasets = 0
    
    # Run cross-validation for airlines dataset with traditional models
    print("\nRunning cross-validation for airlines dataset...")
    airlines_cv_results = run_cross_validation(
        datasets['airlines'],
        'airlines',
        traditional_models,
        n_splits
    )
    completed_datasets += 1
    print(f"\nProgress: {completed_datasets}/{total_datasets} datasets completed")
    
    # Run cross-validation for amazon dataset with traditional models
    print("\nRunning cross-validation for amazon dataset...")
    amazon_cv_results = run_cross_validation(
        datasets['amazon'],
        'amazon',
        traditional_models,
        n_splits
    )
    completed_datasets += 1
    print(f"\nProgress: {completed_datasets}/{total_datasets} datasets completed")
    
    # Run cross-validation for yelp dataset with traditional models
    print("\nRunning cross-validation for yelp dataset...")
    yelp_cv_results = run_cross_validation(
        datasets['yelp'],
        'yelp',
        traditional_models,
        n_splits
    )
    completed_datasets += 1
    print(f"\nProgress: {completed_datasets}/{total_datasets} datasets completed")
    
    # Optional: Run cross-validation for a small subset of airlines data with deep learning models
    # This is optional because deep learning models are resource-intensive
    print("\nRunning cross-validation for a subset of airlines dataset with deep learning models...")
    
    # Load airlines data
    X_airlines, y_airlines = load_data(datasets['airlines'])
    
    # Take a small subset (e.g., 1000 samples) for deep learning models
    subset_size = 1000
    X_subset = X_airlines.iloc[:subset_size]
    y_subset = y_airlines.iloc[:subset_size]
    
    # Save subset to a temporary file
    subset_df = pd.DataFrame({'text': X_subset, 'score': y_subset})
    subset_path = 'data/airlines_subset.csv'
    subset_df.to_csv(subset_path, index=False)
    print(f"Subset saved to {subset_path}")
    
    # Run cross-validation on the subset
    airlines_subset_cv_results = run_cross_validation(
        subset_path,
        'airlines_subset',
        deep_learning_models,
        n_splits=3  # Fewer folds for deep learning models
    )
    completed_datasets += 1
    print(f"\nProgress: {completed_datasets}/{total_datasets} datasets completed")
    
    print("\nCross-validation completed for all datasets and models.")
    print("All results are saved in the 'results/cross_validation' directory.") 