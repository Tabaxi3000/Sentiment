import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Import models
from models.logistic_regression import SentimentLogisticRegression
from models.adaboost_sentiment import SentimentAdaBoost
from models.svm_sentiment import SentimentSVM
from models.bert_sentiment import SentimentBERT
from models.lstm_sentiment import SentimentLSTM
from models.roberta_sentiment import SentimentRoBERTa

def load_data(data_path='data/augmented_data.csv'):
    """Load the augmented dataset"""
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_augmentation.py first.")
        return None, None
    
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['score']
    
    print(f"Loaded {len(df)} samples")
    
    return X, y

def optimize_logistic_regression(X_train, y_train, X_test, y_test, n_iter=50):
    """Optimize hyperparameters for Logistic Regression"""
    print("\nOptimizing Logistic Regression hyperparameters...")
    
    # Define the search space
    param_space = {
        'C': Real(0.01, 10.0, prior='log-uniform'),
        'max_iter': Integer(100, 1000),
        'solver': Categorical(['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']),
        'ngram_range': Categorical([(1, 1), (1, 2), (1, 3)]),
        'max_features': Integer(1000, 10000)
    }
    
    # Create a wrapper for the model
    class LogisticRegressionWrapper:
        def __init__(self, C=1.0, max_iter=100, solver='liblinear', ngram_range=(1, 1), max_features=5000):
            self.C = C
            self.max_iter = int(max_iter)
            self.solver = solver
            self.ngram_range = ngram_range
            self.max_features = int(max_features)
            self.model = SentimentLogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver=self.solver,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
        
        def fit(self, X, y):
            self.model.train(X, y)
            return self
        
        def predict(self, X):
            y_pred = []
            for text in X:
                _, score = self.model.predict(text)
                y_pred.append(score)
            return np.array(y_pred)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
    
    # Create the Bayesian search CV
    opt = BayesSearchCV(
        LogisticRegressionWrapper(),
        param_space,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the optimizer
    opt.fit(X_train, y_train)
    
    # Get the best parameters
    print(f"Best parameters: {opt.best_params_}")
    print(f"Best CV score: {opt.best_score_:.4f}")
    
    # Train the model with the best parameters
    best_model = SentimentLogisticRegression(
        C=opt.best_params_['C'],
        max_iter=int(opt.best_params_['max_iter']),
        solver=opt.best_params_['solver'],
        ngram_range=opt.best_params_['ngram_range'],
        max_features=int(opt.best_params_['max_features'])
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
        'best_params': opt.best_params_,
        'best_score': opt.best_score_,
        'test_accuracy': accuracy,
        'cv_results': opt.cv_results_
    }
    
    joblib.dump(results, 'models/optimized/logistic_regression_optimization_results.joblib')
    
    return best_model, results

def optimize_svm(X_train, y_train, X_test, y_test, n_iter=30):
    """Optimize hyperparameters for SVM"""
    print("\nOptimizing SVM hyperparameters...")
    
    # Define the search space
    param_space = {
        'C': Real(0.1, 10.0, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf', 'poly']),
        'gamma': Real(0.001, 1.0, prior='log-uniform'),
        'ngram_range': Categorical([(1, 1), (1, 2), (1, 3)]),
        'max_features': Integer(1000, 10000)
    }
    
    # Create a wrapper for the model
    class SVMWrapper:
        def __init__(self, C=1.0, kernel='linear', gamma='scale', ngram_range=(1, 1), max_features=5000):
            self.C = C
            self.kernel = kernel
            self.gamma = gamma
            self.ngram_range = ngram_range
            self.max_features = int(max_features)
            self.model = SentimentSVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
        
        def fit(self, X, y):
            self.model.train(X, y)
            return self
        
        def predict(self, X):
            y_pred = []
            for text in X:
                _, score = self.model.predict(text)
                y_pred.append(score)
            return np.array(y_pred)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
    
    # Create the Bayesian search CV
    opt = BayesSearchCV(
        SVMWrapper(),
        param_space,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the optimizer
    opt.fit(X_train, y_train)
    
    # Get the best parameters
    print(f"Best parameters: {opt.best_params_}")
    print(f"Best CV score: {opt.best_score_:.4f}")
    
    # Train the model with the best parameters
    best_model = SentimentSVM(
        C=opt.best_params_['C'],
        kernel=opt.best_params_['kernel'],
        gamma=opt.best_params_['gamma'],
        ngram_range=opt.best_params_['ngram_range'],
        max_features=int(opt.best_params_['max_features'])
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
        'best_params': opt.best_params_,
        'best_score': opt.best_score_,
        'test_accuracy': accuracy,
        'cv_results': opt.cv_results_
    }
    
    joblib.dump(results, 'models/optimized/svm_optimization_results.joblib')
    
    return best_model, results

def optimize_adaboost(X_train, y_train, X_test, y_test, n_iter=30):
    """Optimize hyperparameters for AdaBoost"""
    print("\nOptimizing AdaBoost hyperparameters...")
    
    # Define the search space
    param_space = {
        'n_estimators': Integer(50, 300),
        'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
        'ngram_range': Categorical([(1, 1), (1, 2), (1, 3)]),
        'max_features': Integer(1000, 10000)
    }
    
    # Create a wrapper for the model
    class AdaBoostWrapper:
        def __init__(self, n_estimators=100, learning_rate=1.0, ngram_range=(1, 1), max_features=5000):
            self.n_estimators = int(n_estimators)
            self.learning_rate = learning_rate
            self.ngram_range = ngram_range
            self.max_features = int(max_features)
            self.model = SentimentAdaBoost(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
        
        def fit(self, X, y):
            self.model.train(X, y)
            return self
        
        def predict(self, X):
            y_pred = []
            for text in X:
                _, score = self.model.predict(text)
                y_pred.append(score)
            return np.array(y_pred)
        
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
    
    # Create the Bayesian search CV
    opt = BayesSearchCV(
        AdaBoostWrapper(),
        param_space,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the optimizer
    opt.fit(X_train, y_train)
    
    # Get the best parameters
    print(f"Best parameters: {opt.best_params_}")
    print(f"Best CV score: {opt.best_score_:.4f}")
    
    # Train the model with the best parameters
    best_model = SentimentAdaBoost(
        n_estimators=int(opt.best_params_['n_estimators']),
        learning_rate=opt.best_params_['learning_rate'],
        ngram_range=opt.best_params_['ngram_range'],
        max_features=int(opt.best_params_['max_features'])
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
    best_model.save_model('models/optimized/adaboost_optimized.joblib')
    
    # Save the optimization results
    results = {
        'best_params': opt.best_params_,
        'best_score': opt.best_score_,
        'test_accuracy': accuracy,
        'cv_results': opt.cv_results_
    }
    
    joblib.dump(results, 'models/optimized/adaboost_optimization_results.joblib')
    
    return best_model, results

def plot_optimization_results(results, model_name):
    """Plot the optimization results"""
    print(f"\nPlotting optimization results for {model_name}...")
    
    # Create directory for plots
    os.makedirs('results/optimization', exist_ok=True)
    
    # Extract the results
    cv_results = pd.DataFrame(results['cv_results'])
    
    # Plot the convergence
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results['mean_test_score'])
    plt.xlabel('Iteration')
    plt.ylabel('Mean CV Score')
    plt.title(f'{model_name} Optimization Convergence')
    plt.grid(True)
    plt.savefig(f'results/optimization/{model_name.lower()}_convergence.png')
    plt.close()
    
    # Plot the parameter importance
    param_names = [name for name in cv_results.columns if name.startswith('param_')]
    param_values = cv_results[param_names].values
    
    # Create a correlation matrix
    corr_matrix = np.zeros((len(param_names), 1))
    
    for i, param in enumerate(param_names):
        # Convert categorical parameters to numerical
        if cv_results[param].dtype == object:
            # Create dummy variables
            unique_values = cv_results[param].unique()
            for j, value in enumerate(unique_values):
                dummy = (cv_results[param] == value).astype(int)
                corr = np.corrcoef(dummy, cv_results['mean_test_score'])[0, 1]
                corr_matrix[i, 0] = max(corr_matrix[i, 0], abs(corr))
        else:
            corr = np.corrcoef(cv_results[param], cv_results['mean_test_score'])[0, 1]
            corr_matrix[i, 0] = abs(corr)
    
    # Plot the parameter importance
    plt.figure(figsize=(10, 6))
    param_names = [name.replace('param_', '') for name in param_names]
    plt.barh(param_names, corr_matrix[:, 0])
    plt.xlabel('Absolute Correlation with Mean CV Score')
    plt.title(f'{model_name} Parameter Importance')
    plt.tight_layout()
    plt.savefig(f'results/optimization/{model_name.lower()}_parameter_importance.png')
    plt.close()
    
    # Save the results to CSV
    cv_results.to_csv(f'results/optimization/{model_name.lower()}_cv_results.csv', index=False)
    
    print(f"Plots saved to results/optimization/{model_name.lower()}_*.png")

if __name__ == "__main__":
    # Load the augmented dataset
    X, y = load_data()
    
    if X is None or y is None:
        print("Exiting due to data loading error.")
        exit(1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Optimize Logistic Regression
    lr_model, lr_results = optimize_logistic_regression(X_train, y_train, X_test, y_test)
    plot_optimization_results(lr_results, "Logistic Regression")
    
    # Optimize SVM
    svm_model, svm_results = optimize_svm(X_train, y_train, X_test, y_test)
    plot_optimization_results(svm_results, "SVM")
    
    # Optimize AdaBoost
    adaboost_model, adaboost_results = optimize_adaboost(X_train, y_train, X_test, y_test)
    plot_optimization_results(adaboost_results, "AdaBoost")
    
    print("\nHyperparameter optimization completed successfully!") 