import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import os
import torch
from tqdm import tqdm

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM
from models.bert_sentiment import SentimentBERT
from models.lstm_sentiment import SentimentLSTM
from models.roberta_sentiment import SentimentRoBERTa

def load_and_split_data(data_path, test_size=0.2):
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

def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test, dataset_name):
    """Train and evaluate a specified model"""
    start_time = time.time()
    
    # Initialize model
    if model_name.lower() == 'bert':
        # Use pretrained BERT model for faster results
        # Using textattack/bert-base-uncased-SST-2 which is more reliable
        model = SentimentBERT(epochs=2, use_pretrained=True, pretrained_model="textattack/bert-base-uncased-SST-2")
    elif model_name.lower() == 'lstm':
        model = SentimentLSTM(epochs=3)
    elif model_name.lower() == 'roberta':
        # Use pretrained RoBERTa model for faster results
        model = SentimentRoBERTa(epochs=2, use_pretrained=True, pretrained_model="cardiffnlp/twitter-roberta-base-sentiment")
    elif model_name.lower() == 'logistic':
        model = SentimentLogisticRegression()
    elif model_name.lower() == 'adaboost':
        model = SentimentAdaBoost()
    elif model_name.lower() == 'svm':
        model = SentimentSVM()
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Train
    print(f"\nTraining {model_name} model on {dataset_name} dataset...")
    model.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluation Results:")
    report = model.evaluate(X_test, y_test)
    print(report)
    
    # Calculate accuracy
    y_pred = []
    
    # Use tqdm for prediction progress
    print("\nGenerating predictions for test set...")
    for text in tqdm(X_test, desc="Predicting", unit="sample"):
        sentiment, score = model.predict(text)
        y_pred.append(score)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model
    os.makedirs('models/saved', exist_ok=True)
    model_path = f'models/saved/{model_name.lower()}_{dataset_name}_model.joblib'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    training_time = time.time() - start_time
    print(f"Total time: {training_time:.2f} seconds")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'training_time': training_time
    }

def test_predictions(model):
    """Test model with some example texts"""
    test_texts = [
        "This is absolutely amazing!",
        "I really hate this product.",
        "The service was okay, nothing special.",
        "I had a great experience with this airline.",
        "The flight was delayed and the staff was rude.",
        "The food was delicious and the service was excellent.",
        "I would never recommend this to anyone."
    ]
    
    print("\nTesting predictions:")
    for text in test_texts:
        sentiment, score = model.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment} (score: {score})")

def plot_confusion_matrix(cm, model_name, dataset_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    
    # Create directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_{dataset_name}_cm.png')
    plt.close()
    print(f"Confusion matrix saved to results/{model_name}_{dataset_name}_cm.png")

def run_experiment(dataset_path, dataset_name, models_to_train):
    """Run experiment for a specific dataset and models"""
    results = {}
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"{'='*50}\n")
    
    # Train and evaluate models
    for i, model_name in enumerate(models_to_train):
        print(f"\n{'-'*50}")
        print(f"Model: {model_name} ({i+1}/{len(models_to_train)})")
        print(f"{'-'*50}")
        
        result = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test, dataset_name)
        results[model_name] = result
        
        # Plot confusion matrix
        plot_confusion_matrix(result['confusion_matrix'], model_name, dataset_name)
        
        # Test with example texts
        test_predictions(result['model'])
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    return results

def compare_results(all_results):
    """Compare results across models and datasets"""
    # Create comparison table
    comparison = {
        'Model': [],
        'Dataset': [],
        'Accuracy': [],
        'Training Time (s)': []
    }
    
    for dataset_name, dataset_results in all_results.items():
        for model_name, result in dataset_results.items():
            comparison['Model'].append(model_name)
            comparison['Dataset'].append(dataset_name)
            comparison['Accuracy'].append(result['accuracy'])
            comparison['Training Time (s)'].append(result['training_time'])
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    
    # Training time comparison
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='Training Time (s)', hue='Dataset', data=df)
    plt.title('Model Training Time Comparison')
    plt.yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    print("Model comparison saved to results/model_comparison.png")
    
    # Save comparison to CSV
    df.to_csv('results/model_comparison.csv', index=False)
    print("Model comparison data saved to results/model_comparison.csv")
    
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
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Define datasets
    datasets = {
        'airlines': 'data/airlines_data.csv',
        'amazon': 'data/amazon_cells_labelled.txt',
        'yelp': 'data/yelp_labelled.txt'
    }
    
    # Define models to train
    # For traditional ML models
    traditional_models = ['logistic', 'adaboost', 'svm']
    
    # For deep learning models (more resource-intensive)
    deep_learning_models = ['lstm', 'bert', 'roberta']
    
    # Store all results
    all_results = {}
    
    # Track overall progress
    total_experiments = len(datasets) * (len(traditional_models) + len(deep_learning_models))
    completed_experiments = 0
    
    # Run experiments for all datasets with all models
    for dataset_name, dataset_path in datasets.items():
        print(f"\nRunning experiments for {dataset_name} dataset...")
        all_results[dataset_name] = run_experiment(
            dataset_path, 
            dataset_name,
            traditional_models + deep_learning_models
        )
        completed_experiments += len(traditional_models) + len(deep_learning_models)
        print(f"\nProgress: {completed_experiments}/{total_experiments} experiments completed")
    
    # Compare results
    comparison_df = compare_results(all_results)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Create additional visualizations
    plt.figure(figsize=(14, 10))
    
    # Plot accuracy by model and dataset
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=comparison_df)
    plt.title('Model Accuracy Comparison Across Datasets', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset')
    
    # Plot training time by model and dataset
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='Training Time (s)', hue='Dataset', data=comparison_df)
    plt.title('Model Training Time Comparison Across Datasets', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset')
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_model_comparison.png')
    plt.close()
    print("Comprehensive model comparison saved to results/comprehensive_model_comparison.png")
    
    # Create dataset-specific comparisons
    for dataset_name in datasets.keys():
        dataset_df = comparison_df[comparison_df['Dataset'] == dataset_name]
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy for this dataset
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=dataset_df, palette='viridis')
        plt.title(f'Model Accuracy on {dataset_name.capitalize()} Dataset', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Plot training time for this dataset
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Training Time (s)', data=dataset_df, palette='viridis')
        plt.title(f'Training Time on {dataset_name.capitalize()} Dataset', fontsize=12)
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}_model_comparison.png')
        plt.close()
        print(f"Model comparison for {dataset_name} dataset saved to results/{dataset_name}_model_comparison.png")
    
    print("\nAll experiments completed successfully!") 