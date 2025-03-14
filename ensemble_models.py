import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM
from models.bert_sentiment import SentimentBERT
from models.lstm_sentiment import SentimentLSTM
from models.roberta_sentiment import SentimentRoBERTa

class VotingEnsemble:
    """Voting ensemble that combines multiple models"""
    
    def __init__(self, models, weights=None):
        """
        Initialize the voting ensemble
        
        Args:
            models (list): List of model objects
            weights (list, optional): List of weights for each model. Defaults to None (equal weights).
        """
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        
    def predict(self, text):
        """
        Predict the sentiment of a text using weighted voting
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment, score)
        """
        predictions = []
        scores = []
        
        # Get predictions from all models
        for model in self.models:
            sentiment, score = model.predict(text)
            predictions.append(sentiment)
            scores.append(score)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        scores = np.array(scores)
        
        # Weighted voting
        weighted_scores = scores * self.weights
        final_score = np.sum(weighted_scores)
        
        # Determine sentiment based on score
        if final_score >= 0.5:
            final_sentiment = "positive"
        else:
            final_sentiment = "negative"
        
        return final_sentiment, final_score

class StackingEnsemble:
    """Stacking ensemble that uses a meta-model to combine predictions"""
    
    def __init__(self, base_models, meta_model):
        """
        Initialize the stacking ensemble
        
        Args:
            base_models (list): List of base model objects
            meta_model: Meta-model that combines base model predictions
        """
        self.base_models = base_models
        self.meta_model = meta_model
        
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the stacking ensemble
        
        Args:
            X_train (list): Training texts
            y_train (list): Training labels
            X_val (list): Validation texts
            y_val (list): Validation labels
        """
        print("Training base models...")
        
        # Train base models
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}...")
            model.train(X_train, y_train)
        
        # Generate meta-features
        print("Generating meta-features...")
        meta_features = self._generate_meta_features(X_val)
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.train(meta_features, y_val)
        
        print("Stacking ensemble training completed.")
    
    def _generate_meta_features(self, texts):
        """
        Generate meta-features from base models
        
        Args:
            texts (list): Input texts
            
        Returns:
            pandas.DataFrame: Meta-features
        """
        meta_features = []
        
        for text in tqdm(texts, desc="Generating meta-features"):
            features = []
            
            for model in self.base_models:
                _, score = model.predict(text)
                features.append(score)
            
            meta_features.append(features)
        
        return pd.DataFrame(meta_features)
    
    def predict(self, text):
        """
        Predict the sentiment of a text using the stacking ensemble
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment, score)
        """
        # Generate meta-features
        features = []
        
        for model in self.base_models:
            _, score = model.predict(text)
            features.append(score)
        
        # Convert to DataFrame
        meta_features = pd.DataFrame([features])
        
        # Predict using meta-model
        sentiment, score = self.meta_model.predict(meta_features.iloc[0])
        
        return sentiment, score

def load_optimized_models():
    """
    Load optimized models from disk
    
    Returns:
        dict: Dictionary of optimized models
    """
    print("Loading optimized models...")
    
    models = {}
    
    # Check if optimized models exist
    if os.path.exists('models/optimized/logistic_regression_optimized.joblib'):
        print("Loading optimized Logistic Regression model...")
        lr_model = SentimentLogisticRegression()
        lr_model.load_model('models/optimized/logistic_regression_optimized.joblib')
        models['logistic_regression'] = lr_model
    
    if os.path.exists('models/optimized/svm_optimized.joblib'):
        print("Loading optimized SVM model...")
        svm_model = SentimentSVM()
        svm_model.load_model('models/optimized/svm_optimized.joblib')
        models['svm'] = svm_model
    
    if os.path.exists('models/optimized/adaboost_optimized.joblib'):
        print("Loading optimized AdaBoost model...")
        adaboost_model = SentimentAdaBoost()
        adaboost_model.load_model('models/optimized/adaboost_optimized.joblib')
        models['adaboost'] = adaboost_model
    
    # If no optimized models found, train default models
    if not models:
        print("No optimized models found. Training default models...")
        
        # Load data
        df = pd.read_csv('data/augmented_data.csv')
        X = df['text']
        y = df['score']
        
        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        print("Training Logistic Regression model...")
        lr_model = SentimentLogisticRegression()
        lr_model.train(X_train, y_train)
        models['logistic_regression'] = lr_model
        
        print("Training SVM model...")
        svm_model = SentimentSVM()
        svm_model.train(X_train, y_train)
        models['svm'] = svm_model
        
        print("Training AdaBoost model...")
        adaboost_model = SentimentAdaBoost()
        adaboost_model.train(X_train, y_train)
        models['adaboost'] = adaboost_model
    
    return models

def create_voting_ensemble(models, weights=None):
    """
    Create a voting ensemble from the given models
    
    Args:
        models (dict): Dictionary of models
        weights (dict, optional): Dictionary of weights for each model. Defaults to None.
        
    Returns:
        VotingEnsemble: Voting ensemble
    """
    model_list = list(models.values())
    
    if weights is not None:
        weight_list = [weights.get(name, 1) for name in models.keys()]
    else:
        weight_list = None
    
    return VotingEnsemble(model_list, weight_list)

def create_stacking_ensemble(models, meta_model_type='logistic_regression'):
    """
    Create a stacking ensemble from the given models
    
    Args:
        models (dict): Dictionary of models
        meta_model_type (str, optional): Type of meta-model. Defaults to 'logistic_regression'.
        
    Returns:
        StackingEnsemble: Stacking ensemble
    """
    base_models = list(models.values())
    
    # Create meta-model
    if meta_model_type == 'logistic_regression':
        meta_model = SentimentLogisticRegression()
    elif meta_model_type == 'svm':
        meta_model = SentimentSVM()
    elif meta_model_type == 'adaboost':
        meta_model = SentimentAdaBoost()
    else:
        raise ValueError(f"Unsupported meta-model type: {meta_model_type}")
    
    return StackingEnsemble(base_models, meta_model)

def evaluate_ensemble(ensemble, X_test, y_test):
    """
    Evaluate the ensemble on the test set
    
    Args:
        ensemble: Ensemble model
        X_test (list): Test texts
        y_test (list): Test labels
        
    Returns:
        dict: Evaluation results
    """
    print("Evaluating ensemble...")
    
    y_pred = []
    scores = []
    
    for text in tqdm(X_test, desc="Evaluating"):
        sentiment, score = ensemble.predict(text)
        y_pred.append(1 if sentiment == "positive" else 0)
        scores.append(score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Create directory for results
    os.makedirs('results/ensemble', exist_ok=True)
    plt.savefig('results/ensemble/confusion_matrix.png')
    plt.close()
    
    # Return results
    results = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'scores': scores
    }
    
    return results

def compare_models(models, ensemble_results, X_test, y_test):
    """
    Compare individual models with the ensemble
    
    Args:
        models (dict): Dictionary of individual models
        ensemble_results (dict): Results of the ensemble
        X_test (list): Test texts
        y_test (list): Test labels
    """
    print("Comparing models...")
    
    # Evaluate individual models
    model_results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        y_pred = []
        
        for text in tqdm(X_test, desc=f"Evaluating {name}"):
            sentiment, _ = model.predict(text)
            y_pred.append(1 if sentiment == "positive" else 0)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        model_results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Compare accuracies
    accuracies = {name: results['accuracy'] for name, results in model_results.items()}
    accuracies['ensemble'] = ensemble_results['accuracy']
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/ensemble/model_comparison.png')
    plt.close()
    
    # Save results
    comparison = {
        'individual_models': model_results,
        'ensemble': ensemble_results
    }
    
    joblib.dump(comparison, 'results/ensemble/model_comparison.joblib')
    
    print("Model comparison completed.")

def main():
    """Main function"""
    print("Starting ensemble model training and evaluation...")
    
    # Load data
    print("Loading data...")
    
    if not os.path.exists('data/augmented_data.csv'):
        print("Error: Augmented data not found. Please run data_augmentation.py first.")
        return
    
    df = pd.read_csv('data/augmented_data.csv')
    X = df['text']
    y = df['score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Load optimized models
    models = load_optimized_models()
    
    # Create voting ensemble
    print("\nCreating voting ensemble...")
    weights = {
        'logistic_regression': 1,
        'svm': 2,  # Give more weight to SVM as it typically performs better
        'adaboost': 1
    }
    
    voting_ensemble = create_voting_ensemble(models, weights)
    
    # Evaluate voting ensemble
    print("\nEvaluating voting ensemble...")
    voting_results = evaluate_ensemble(voting_ensemble, X_test, y_test)
    
    # Create stacking ensemble
    print("\nCreating stacking ensemble...")
    stacking_ensemble = create_stacking_ensemble(models, meta_model_type='svm')
    
    # Train stacking ensemble
    print("\nTraining stacking ensemble...")
    stacking_ensemble.train(X_train, y_train, X_val, y_val)
    
    # Evaluate stacking ensemble
    print("\nEvaluating stacking ensemble...")
    stacking_results = evaluate_ensemble(stacking_ensemble, X_test, y_test)
    
    # Compare models
    print("\nComparing models...")
    compare_models(models, stacking_results, X_test, y_test)
    
    # Save ensembles
    print("\nSaving ensembles...")
    os.makedirs('models/ensemble', exist_ok=True)
    
    # We can't directly save the ensemble objects as they contain model objects
    # Instead, save the model paths and weights/configuration
    voting_config = {
        'model_paths': {name: f'models/optimized/{name}_optimized.joblib' for name in models.keys()},
        'weights': weights
    }
    
    stacking_config = {
        'base_model_paths': {name: f'models/optimized/{name}_optimized.joblib' for name in models.keys()},
        'meta_model_type': 'svm'
    }
    
    joblib.dump(voting_config, 'models/ensemble/voting_ensemble_config.joblib')
    joblib.dump(stacking_config, 'models/ensemble/stacking_ensemble_config.joblib')
    
    print("\nEnsemble models training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 