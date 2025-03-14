# Sentiment Analysis Project

This project implements various machine learning and deep learning models for sentiment analysis on multiple datasets.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models](#models)
- [Advanced Features](#advanced-features)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

This sentiment analysis project implements and compares various machine learning and deep learning approaches for classifying text as positive or negative. The project includes traditional machine learning models like Logistic Regression, SVM, and AdaBoost, as well as deep learning models like LSTM, BERT, and RoBERTa.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tabaxi3000/Sentiment.git
cd Sentiment
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv sentiment_env
source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate
pip install -r requirements.txt
```

## Datasets

The project uses several datasets for sentiment analysis:

1. **Airlines Dataset**: Contains tweets about airlines with sentiment labels.
2. **Amazon Reviews**: Product reviews from Amazon with sentiment labels.
3. **Yelp Reviews**: Business reviews from Yelp with sentiment labels.
4. **IMDB Reviews**: Movie reviews from IMDB with sentiment labels.

### Dataset Analysis

Each dataset presents unique challenges for sentiment analysis:

- **Airlines Dataset**: Contains short texts (tweets) with airline-specific terminology and often sarcasm.
- **Amazon Reviews**: Product reviews vary in length and contain product-specific terminology.
- **Yelp Reviews**: Business reviews often contain location-specific references and service-related terminology.
- **IMDB Reviews**: Movie reviews tend to be longer and contain movie-specific terminology and references.

## Models

The project implements the following models:

### Traditional Machine Learning Models
- **Logistic Regression**: A simple baseline model using TF-IDF features.
- **Support Vector Machine (SVM)**: A powerful model for text classification using TF-IDF features.
- **AdaBoost**: An ensemble method that combines multiple weak classifiers.

### Deep Learning Models
- **LSTM**: A recurrent neural network architecture suitable for sequential data like text.
- **BERT**: A transformer-based model pre-trained on a large corpus of text.
- **RoBERTa**: An optimized version of BERT with improved training methodology.

## Advanced Features

### Data Augmentation

The project includes a data augmentation module (`data_augmentation.py`) that combines all datasets and implements various text augmentation techniques:

- **Synonym Replacement**: Replaces random words with their synonyms.
- **Random Insertion**: Inserts synonyms of random words at random positions.
- **Random Swap**: Swaps random words in the text.
- **Random Deletion**: Randomly deletes words from the text.
- **BERT-based Augmentation**: Uses BERT to generate contextually similar sentences.

To run data augmentation:
```bash
python data_augmentation.py
```

This will create an augmented dataset at `data/augmented_data.csv`.

### Advanced Hyperparameter Tuning

The project includes an advanced hyperparameter tuning module (`advanced_hyperparameter_tuning.py`) that uses Bayesian optimization to find the best hyperparameters for each model:

- **Logistic Regression**: Optimizes C, max_iter, solver, ngram_range, and max_features.
- **SVM**: Optimizes C, kernel, gamma, ngram_range, and max_features.
- **AdaBoost**: Optimizes n_estimators, learning_rate, ngram_range, and max_features.

To run hyperparameter tuning:
```bash
python advanced_hyperparameter_tuning.py
```

This will save the optimized models to `models/optimized/` and the optimization results to `results/optimization/`.

### Ensemble Methods

The project includes an ensemble methods module (`ensemble_models.py`) that combines multiple models for better performance:

- **Voting Ensemble**: Combines predictions from multiple models using weighted voting.
- **Stacking Ensemble**: Uses a meta-model to combine predictions from base models.

To run ensemble methods:
```bash
python ensemble_models.py
```

This will save the ensemble models to `models/ensemble/` and the evaluation results to `results/ensemble/`.

## Usage

### Basic Usage

To run the main experiment:
```bash
python main.py
```

This will train and evaluate all models on all datasets.

### Cross-Validation

To run cross-validation:
```bash
python cross_validation.py
```

This will perform k-fold cross-validation on all models.

### Hyperparameter Tuning

To run basic hyperparameter tuning:
```bash
python hyperparameter_tuning.py
```

This will perform grid search for hyperparameter tuning.

### Advanced Usage

For advanced usage with data augmentation, advanced hyperparameter tuning, and ensemble methods:

1. Run data augmentation:
```bash
python data_augmentation.py
```

2. Run advanced hyperparameter tuning:
```bash
python advanced_hyperparameter_tuning.py
```

3. Run ensemble methods:
```bash
python ensemble_models.py
```

## Results

The project evaluates models based on accuracy, precision, recall, and F1-score. The results are saved to the `results/` directory.

### Model Performance

Based on our experiments, the SVM model generally performs the best among traditional machine learning models, while BERT and RoBERTa perform the best among deep learning models. The ensemble methods typically outperform individual models.

### Performance Analysis

- **Traditional ML Models**: These models are faster to train and can perform well on smaller datasets. SVM typically achieves the best performance among these models.
- **Deep Learning Models**: These models require more computational resources but can capture more complex patterns in the text. BERT and RoBERTa typically outperform LSTM.
- **Ensemble Methods**: Combining multiple models can lead to better performance than individual models. The stacking ensemble typically outperforms the voting ensemble.

## Future Improvements

Potential improvements for the project include:

- **More Advanced Augmentation**: Implement more advanced text augmentation techniques.
- **Model Distillation**: Distill knowledge from large models to smaller, more efficient models.
- **Multi-task Learning**: Train models on multiple related tasks to improve performance.
- **Explainability**: Implement methods to explain model predictions.
- **Deployment**: Create a web application for real-time sentiment analysis.

## Features

- **GPU Acceleration**: Automatic detection and utilization of GPU for faster training
- **Progress Tracking**: Visual progress bars for training, evaluation, and hyperparameter tuning
- **Cross-Validation**: K-fold cross-validation for reliable model assessment
- **Hyperparameter Tuning**: Grid search for finding optimal model parameters
- **Visualization**: Comprehensive visualization of model performance metrics
- **Early Stopping**: Prevents overfitting by stopping training when performance plateaus
- **Model Persistence**: Save and load trained models for future use

## Project Structure

```
.
├── data/                      # Dataset directory
│   ├── airlines_data.csv      # Airlines sentiment dataset
│   ├── amazon_cells_labelled.txt  # Amazon product reviews
│   └── yelp_labelled.txt      # Yelp restaurant reviews
├── models/                    # Model implementations
│   ├── adaboost_sentiment.py  # AdaBoost model
│   ├── bert_sentiment.py      # BERT model
│   ├── logistic_regression.py # Logistic Regression model
│   ├── lstm_sentiment.py      # LSTM model
│   ├── roberta_sentiment.py   # RoBERTa model
│   └── svm_sentiment.py       # SVM model
├── results/                   # Results directory
│   ├── cross_validation/      # Cross-validation results
│   └── tuning/                # Hyperparameter tuning results
├── main.py                    # Main script for running experiments
├── cross_validation.py        # Script for cross-validation
├── hyperparameter_tuning.py   # Script for hyperparameter tuning
└── requirements.txt           # Python dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing pre-trained transformer models
- scikit-learn for traditional machine learning implementations
- PyTorch for deep learning framework

## Results Location

- Model comparison visualizations: `results/`
- Trained models: `models/saved/`
- Hyperparameter tuning results: `results/tuning/`
- Cross-validation results: `results/cross_validation/`

## Potential Improvements for Deep Learning Models

While traditional ML models might outperform deep learning models in our current setup, several strategies could potentially improve the performance of deep learning models:

### 1. Fine-tuning Pretrained Models

- **Current Approach**: Using pretrained models directly without task-specific fine-tuning
- **Improvement**: Fine-tune BERT and RoBERTa on our specific datasets
- **Expected Benefit**: Better adaptation to domain-specific language and sentiment patterns

### 2. Data Augmentation

- **Current Approach**: Using the original datasets as-is
- **Improvement**: Augment training data through techniques like:
  - Synonym replacement
  - Random insertion/deletion/swap of words
  - Back-translation
  - Using language models to generate similar examples
- **Expected Benefit**: Increased training data size, improved generalization

### 3. Hyperparameter Optimization

- **Current Approach**: Using default or minimally tuned hyperparameters
- **Improvement**: Systematic hyperparameter tuning through:
  - Grid search
  - Random search
  - Bayesian optimization
- **Expected Benefit**: Models optimized for our specific tasks and datasets

### 4. Model Architecture Modifications

- **Current Approach**: Using standard BERT/RoBERTa/LSTM architectures
- **Improvement**: Customize model architectures:
  - Add task-specific layers
  - Modify attention mechanisms
  - Implement ensemble methods
- **Expected Benefit**: Architectures better suited to sentiment analysis tasks

### 5. Domain-Specific Pretraining

- **Current Approach**: Using general-purpose pretrained models
- **Improvement**: Further pretrain models on domain-specific corpora:
  - Airline reviews for the airlines dataset
  - Product reviews for the Amazon dataset
  - Business reviews for the Yelp dataset
- **Expected Benefit**: Better understanding of domain-specific language

### 6. Advanced Preprocessing

- **Current Approach**: Basic text preprocessing
- **Improvement**: Implement advanced preprocessing:
  - Better handling of domain-specific terms
  - Special handling of negations
  - Entity recognition and normalization
- **Expected Benefit**: Higher quality input data for models

## Ensemble Methods for Improved Performance

Combining multiple models through ensemble methods could potentially improve overall performance beyond what any single model can achieve:

### 1. Voting Ensemble

- **Approach**: Combine predictions from multiple models through majority voting
- **Implementation**: 
  - Hard voting: Select the class predicted by the majority of models
  - Soft voting: Select the class with the highest average probability
- **Benefits**:
  - Reduces variance and bias
  - More robust predictions
  - Can combine strengths of different model types

### 2. Stacking Ensemble

- **Approach**: Train a meta-model to combine predictions from base models
- **Implementation**:
  - Use predictions from all models as features for a meta-model
  - Meta-model learns optimal weighting of base models
- **Benefits**:
  - Can learn complex relationships between model predictions
  - Often outperforms individual models
  - Adapts to strengths and weaknesses of base models

### 3. Model-Specific Ensembles

- **Approach**: Create ensembles of the same model type with different configurations
- **Implementation**:
  - Train multiple instances of BERT/RoBERTa with different hyperparameters
  - Train multiple traditional ML models with different feature sets
- **Benefits**:
  - Reduces overfitting
  - Improves generalization
  - Can capture different aspects of the data

### 4. Dataset-Specific Ensembles

- **Approach**: Create specialized ensembles for each dataset
- **Implementation**:
  - Select the best-performing models for each specific dataset
  - Weight models based on their performance on the specific dataset
- **Benefits**:
  - Optimized for domain-specific characteristics
  - Can address dataset-specific challenges

### 5. Feature-Level Ensemble

- **Approach**: Combine features from different models before classification
- **Implementation**:
  - Extract embeddings/features from different models
  - Concatenate or combine features
  - Train a classifier on the combined features
- **Benefits**:
  - Leverages different feature representations
  - Can capture both lexical and semantic information

## Performance Summary

Based on our evaluation across multiple datasets, we observe the following patterns:

### Model Accuracy Comparison

| Model | Airlines | Amazon | Yelp | Average |
|-------|----------|--------|------|---------|
| SVM | **0.909** | **0.865** | **0.815** | **0.863** |
| Logistic Regression | 0.901 | 0.820 | 0.795 | 0.839 |
| RoBERTa | 0.870 | - | - | - |
| AdaBoost | 0.871 | 0.770 | 0.715 | 0.785 |
| BERT | 0.864 | - | - | - |
| LSTM | 0.802 | - | - | - |

### Key Observations

1. **SVM Consistently Outperforms Other Models**:
   - Highest accuracy across all three datasets
   - Particularly effective on the Amazon dataset (0.865)
   - Balances good performance with reasonable training time

2. **Traditional ML vs. Deep Learning**:
   - Traditional ML models (especially SVM and Logistic Regression) outperform deep learning models
   - This is likely due to the characteristics of the datasets (small size, clear lexical patterns)
   - Using pretrained models without fine-tuning limits the potential of BERT and RoBERTa

3. **Training Efficiency**:
   - Logistic Regression is extremely fast (0.08-1.10 seconds)
   - Deep learning models are significantly slower (BERT/RoBERTa: ~1523 seconds)
   - The performance-to-time ratio strongly favors traditional ML models

4. **Dataset-Specific Performance**:
   - All models perform best on the Airlines dataset
   - The Yelp dataset appears to be the most challenging
   - This suggests differences in the complexity and clarity of sentiment signals across datasets

### Recommendations

1. **For Production Deployment**:
   - SVM offers the best balance of accuracy and efficiency
   - Logistic Regression is a good alternative when speed is critical

2. **For Further Research**:
   - Implement fine-tuning for BERT and RoBERTa models
   - Explore ensemble methods combining traditional ML and deep learning approaches
   - Investigate domain-specific pretraining for deep learning models
