#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit App for Sentiment Analysis

This app provides a user-friendly web interface for sentiment analysis
using the trained models from the combined dataset.

Author: Sentiment Analysis Team
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from io import StringIO

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
            confidence: Confidence score between 0 and 1
        """
        votes = []
        individual_predictions = {}
        
        for i, model in enumerate(self.models):
            sentiment, score = model.predict(text)
            votes.append((score, self.weights[i]))
            
            # Store individual model predictions
            if isinstance(model, SentimentLogisticRegression):
                individual_predictions["Logistic Regression"] = sentiment
            elif isinstance(model, SentimentSVM):
                individual_predictions["SVM"] = sentiment
            elif isinstance(model, SentimentAdaBoost):
                individual_predictions["AdaBoost"] = sentiment
        
        # Calculate weighted vote
        weighted_sum = sum(vote * weight for vote, weight in votes)
        total_weight = sum(self.weights)
        
        # Calculate confidence (how strong the majority is)
        confidence = abs((weighted_sum / total_weight) - 0.5) * 2  # Scale to 0-1
        
        # Determine final prediction
        if weighted_sum / total_weight >= 0.5:
            return "positive", 1, confidence, individual_predictions
        else:
            return "negative", 0, confidence, individual_predictions

def load_model(model_type, model_dir='models/combined'):
    """
    Load a trained model
    
    Args:
        model_type: Type of model to load ('lr', 'svm', 'adaboost', 'ensemble', 'weighted')
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
        st.error(f"Error: Unknown model type '{model_type}'")
        st.info(f"Available models: {', '.join(list(model_paths.keys()) + ['ensemble', 'weighted'])}")
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
                st.error("Error: No models found for ensemble")
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
                st.error(f"Error: Model file '{model_paths[model_type]}' not found")
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
        st.error(f"Error loading model: {str(e)}")
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
        confidence: Confidence score (if available)
        individual_predictions: Individual model predictions (if ensemble)
    """
    try:
        if isinstance(model, VotingEnsemble):
            sentiment, score, confidence, individual_predictions = model.predict(text)
            return sentiment, score, confidence, individual_predictions
        else:
            sentiment, score = model.predict(text)
            return sentiment, score, None, None
    except Exception as e:
        st.error(f"Error predicting sentiment: {str(e)}")
        return "error", -1, None, None

def predict_batch(texts, model):
    """
    Predict sentiment for a batch of texts
    
    Args:
        texts: List of texts
        model: Trained model
        
    Returns:
        results: List of (text, sentiment, score, confidence) tuples
    """
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        if isinstance(model, VotingEnsemble):
            sentiment, score, confidence, individual_predictions = model.predict(text)
            results.append((text, sentiment, score, confidence, individual_predictions))
        else:
            sentiment, score = model.predict(text)
            results.append((text, sentiment, score, None, None))
        
        # Update progress
        progress = (i + 1) / len(texts)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {i+1}/{len(texts)} texts")
    
    status_text.text("Processing complete!")
    return results

def set_page_config():
    """Set Streamlit page configuration"""
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-high {
        color: #4CAF50;
    }
    .confidence-medium {
        color: #FF9800;
    }
    .confidence-low {
        color: #F44336;
    }
    .model-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    set_page_config()
    
    st.markdown("<h1 class='main-header'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Analyze the sentiment of text using machine learning models</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_options = {
        "weighted": "Weighted Ensemble (Recommended)",
        "ensemble": "Equal Weights Ensemble",
        "lr": "Logistic Regression",
        "svm": "Support Vector Machine",
        "adaboost": "AdaBoost"
    }
    
    selected_model_key = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Load the selected model
    model = load_model(selected_model_key)
    
    if model is None:
        st.error("Failed to load model. Please check the model files.")
        return
    
    # Model information
    with st.sidebar.expander("Model Information"):
        if selected_model_key == "weighted":
            st.write("**Weighted Ensemble**")
            st.write("Combines predictions from multiple models with different weights:")
            st.write("- Logistic Regression (weight: 1)")
            st.write("- SVM (weight: 2)")
            st.write("- AdaBoost (weight: 1)")
            st.write("Accuracy: 84.18%")
        elif selected_model_key == "ensemble":
            st.write("**Equal Weights Ensemble**")
            st.write("Combines predictions from multiple models with equal weights:")
            st.write("- Logistic Regression")
            st.write("- SVM")
            st.write("- AdaBoost")
            st.write("Accuracy: 83.82%")
        elif selected_model_key == "lr":
            st.write("**Logistic Regression**")
            st.write("A linear model for binary classification.")
            st.write("Accuracy: 83.09%")
        elif selected_model_key == "svm":
            st.write("**Support Vector Machine**")
            st.write("A powerful classification algorithm that finds the optimal hyperplane.")
            st.write("Accuracy: 84.00%")
        elif selected_model_key == "adaboost":
            st.write("**AdaBoost**")
            st.write("An ensemble method that combines multiple weak classifiers.")
            st.write("Accuracy: 79.27%")
    
    # About section
    with st.sidebar.expander("About"):
        st.write("""
        This app uses machine learning models trained on a combined dataset of:
        - Amazon product reviews
        - Yelp reviews
        - IMDB movie reviews
        - Airline tweets
        
        The models were trained to classify text as either positive or negative sentiment.
        """)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "Examples"])
    
    # Single Text Analysis
    with tab1:
        st.markdown("<h2 class='sub-header'>Analyze Text Sentiment</h2>", unsafe_allow_html=True)
        
        text_input = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Analyze Sentiment", key="analyze_single"):
            if not text_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    # Add a small delay to show the spinner
                    time.sleep(0.5)
                    
                    # Predict sentiment
                    if isinstance(model, VotingEnsemble):
                        sentiment, score, confidence, individual_predictions = predict_sentiment(text_input, model)
                    else:
                        sentiment, score, _, _ = predict_sentiment(text_input, model)
                        confidence = None
                        individual_predictions = None
                    
                    # Display result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Result")
                        if sentiment == "positive":
                            st.markdown(f"<div class='positive'>Positive Sentiment</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='negative'>Negative Sentiment</div>", unsafe_allow_html=True)
                        
                        if confidence is not None:
                            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                            st.markdown(f"<div class='{confidence_class}'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
                    
                    # Display individual model predictions if using ensemble
                    if individual_predictions:
                        with col2:
                            st.markdown("### Individual Model Predictions")
                            for model_name, pred in individual_predictions.items():
                                pred_class = "positive" if pred == "positive" else "negative"
                                st.markdown(f"<div class='model-card'><b>{model_name}:</b> <span class='{pred_class}'>{pred}</span></div>", unsafe_allow_html=True)
    
    # Batch Analysis
    with tab2:
        st.markdown("<h2 class='sub-header'>Batch Analysis</h2>", unsafe_allow_html=True)
        st.write("Analyze multiple texts at once by uploading a file or entering multiple lines.")
        
        batch_method = st.radio("Input Method:", ["Text Input", "File Upload"])
        
        texts_to_analyze = []
        
        if batch_method == "Text Input":
            batch_text = st.text_area("Enter multiple texts (one per line):", height=200)
            if batch_text:
                texts_to_analyze = [line.strip() for line in batch_text.split("\n") if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload a text file (one text per line):", type=["txt", "csv"])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts_to_analyze = df['text'].tolist()
                        else:
                            texts_to_analyze = df.iloc[:, 0].tolist()
                    else:
                        content = uploaded_file.getvalue().decode("utf-8")
                        texts_to_analyze = [line.strip() for line in content.split("\n") if line.strip()]
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        if st.button("Analyze Batch", key="analyze_batch") and texts_to_analyze:
            if len(texts_to_analyze) > 100:
                st.warning(f"You've provided {len(texts_to_analyze)} texts. Processing may take some time.")
            
            with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                # Predict sentiment for all texts
                results = predict_batch(texts_to_analyze, model)
                
                # Create DataFrame from results
                if isinstance(model, VotingEnsemble):
                    df_results = pd.DataFrame([
                        {"Text": text, "Sentiment": sentiment, "Score": score, "Confidence": conf}
                        for text, sentiment, score, conf, _ in results
                    ])
                else:
                    df_results = pd.DataFrame([
                        {"Text": text, "Sentiment": sentiment, "Score": score}
                        for text, sentiment, score, _, _ in results
                    ])
                
                # Display results
                st.markdown("### Results")
                st.dataframe(df_results)
                
                # Display statistics
                st.markdown("### Statistics")
                positive_count = sum(1 for _, sentiment, _, _, _ in results if sentiment == "positive")
                negative_count = len(results) - positive_count
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Positive Texts", positive_count, f"{positive_count/len(results)*100:.1f}%")
                    st.metric("Negative Texts", negative_count, f"{negative_count/len(results)*100:.1f}%")
                
                with col2:
                    # Create a pie chart
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie([positive_count, negative_count], labels=["Positive", "Negative"], 
                           autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
                    ax.set_title("Sentiment Distribution")
                    st.pyplot(fig)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
    
    # Examples
    with tab3:
        st.markdown("<h2 class='sub-header'>Example Texts</h2>", unsafe_allow_html=True)
        st.write("Click on any example to analyze its sentiment.")
        
        examples = [
            "This product exceeded my expectations. The quality is excellent and the price is reasonable.",
            "The customer service was terrible. I waited for hours and no one helped me.",
            "The movie was okay, not great but not terrible either.",
            "I absolutely love this restaurant! The food is delicious and the staff is friendly.",
            "This flight was delayed by 3 hours and the staff was rude and unhelpful.",
            "The hotel room was clean and comfortable, but the Wi-Fi was slow.",
            "I would never recommend this service to anyone. Complete waste of money.",
            "The concert was amazing! The band played all my favorite songs."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.text_area("Selected Example:", example, height=100, key=f"example_text_{i}")
                
                with st.spinner("Analyzing sentiment..."):
                    # Add a small delay to show the spinner
                    time.sleep(0.5)
                    
                    # Predict sentiment
                    if isinstance(model, VotingEnsemble):
                        sentiment, score, confidence, individual_predictions = predict_sentiment(example, model)
                    else:
                        sentiment, score, _, _ = predict_sentiment(example, model)
                        confidence = None
                        individual_predictions = None
                    
                    # Display result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Result")
                        if sentiment == "positive":
                            st.markdown(f"<div class='positive'>Positive Sentiment</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='negative'>Negative Sentiment</div>", unsafe_allow_html=True)
                        
                        if confidence is not None:
                            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                            st.markdown(f"<div class='{confidence_class}'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
                    
                    # Display individual model predictions if using ensemble
                    if individual_predictions:
                        with col2:
                            st.markdown("### Individual Model Predictions")
                            for model_name, pred in individual_predictions.items():
                                pred_class = "positive" if pred == "positive" else "negative"
                                st.markdown(f"<div class='model-card'><b>{model_name}:</b> <span class='{pred_class}'>{pred}</span></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 