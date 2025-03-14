# Sentiment Analysis Project

This project implements and compares various machine learning and deep learning models for sentiment analysis on multiple datasets.

## Overview

This project provides a framework for sentiment analysis using both traditional machine learning models and state-of-the-art deep learning approaches. It includes:

- Data preprocessing and feature extraction
- Model training and evaluation
- Cross-validation for robust performance assessment
- Hyperparameter tuning for model optimization
- Visualization of results for easy comparison

## Models Implemented

1. **Traditional Machine Learning Models**:
   - Logistic Regression with TF-IDF features
   - AdaBoost with Decision Trees
   - Support Vector Machine (SVM) with TF-IDF features

2. **Deep Learning Models**:
   - Long Short-Term Memory (LSTM) network
   - BERT (Bidirectional Encoder Representations from Transformers)
   - RoBERTa (Robustly Optimized BERT Pretraining Approach)

## Datasets

The project evaluates the models on three datasets:
- **Airlines**: Customer reviews of airlines (positive/negative sentiment)
- **Amazon**: Product reviews from Amazon (positive/negative sentiment)
- **Yelp**: Business reviews from Yelp (positive/negative sentiment)

### Dataset Characteristics

#### Airlines Dataset
- **Source**: Twitter data about airline customer experiences
- **Content**: Short tweets about airline experiences, often containing specific complaints or praise
- **Language**: Often contains airline-specific terminology and abbreviations
- **Challenges**: 
  - Short text length limits contextual information
  - Contains Twitter-specific language patterns
  - Often includes mentions, hashtags, and other Twitter-specific elements
  - Highly imbalanced (more negative than positive reviews)

#### Amazon Dataset
- **Source**: Amazon product reviews
- **Content**: Product reviews covering various categories
- **Language**: Consumer-focused, product-specific terminology
- **Challenges**:
  - Product-specific vocabulary
  - Reviews often focus on specific product features
  - Sentiment can be mixed within a single review

#### Yelp Dataset
- **Source**: Yelp business reviews
- **Content**: Reviews of various businesses (restaurants, services, etc.)
- **Language**: Service and experience-focused terminology
- **Challenges**:
  - Domain-specific vocabulary (restaurant terms, service descriptions)
  - Often contains nuanced opinions
  - Can include multiple aspects of a business in a single review

### Why Traditional ML Models Might Outperform Deep Learning on These Datasets

1. **Text Length and Structure**:
   - These datasets often contain short, direct statements of sentiment
   - Traditional ML models with bag-of-words/TF-IDF features can effectively capture sentiment from key words
   - Deep learning models might not have enough context to leverage their strengths

2. **Dataset Size**:
   - The datasets are relatively small compared to what deep learning models typically need
   - Traditional ML models can learn effectively from smaller datasets

3. **Domain Specificity**:
   - Each dataset contains domain-specific language
   - Pretrained models might not be optimized for these specific domains
   - Traditional ML models learn directly from the domain-specific data

4. **Clear Sentiment Indicators**:
   - These datasets often contain clear lexical indicators of sentiment
   - Words like "great", "terrible", "love", "hate" are strong predictors
   - TF-IDF features can effectively capture these indicators

## Performance Analysis

### Why Different Models Perform Differently

#### Traditional ML Models vs. Deep Learning Models

1. **Data Size Requirements**:
   - Deep learning models (BERT, RoBERTa, LSTM) typically require large amounts of training data to outperform simpler models.
   - Traditional ML models (Logistic Regression, SVM, AdaBoost) can perform well with smaller datasets.

2. **Feature Representation**:
   - Traditional ML models use TF-IDF features that capture important word frequencies.
   - Deep learning models learn contextual representations that can capture more complex patterns.
   - For datasets with clear lexical patterns, TF-IDF features might be sufficient.

3. **Training Efficiency**:
   - Traditional ML models train much faster and require fewer computational resources.
   - Deep learning models, especially transformer-based models like BERT and RoBERTa, are computationally intensive.

4. **Pretrained Knowledge**:
   - BERT and RoBERTa leverage pretrained knowledge from large corpora.
   - When using pretrained models without fine-tuning, performance depends on how well the pretraining aligns with the target task.

#### Specific Model Characteristics

1. **Logistic Regression**:
   - Simple and effective for linearly separable data.
   - Works well when specific words are strong indicators of sentiment.
   - Less prone to overfitting on small datasets.

2. **SVM**:
   - Effective in high-dimensional spaces (like text data).
   - Can capture non-linear decision boundaries with appropriate kernels.
   - Generally performs well on text classification tasks.

3. **AdaBoost**:
   - Focuses on difficult-to-classify examples.
   - Can capture more complex patterns than logistic regression.
   - May be sensitive to noisy data.

4. **LSTM**:
   - Captures sequential patterns and long-range dependencies in text.
   - Requires more data to learn effectively.
   - Performance depends heavily on hyperparameter settings.

5. **BERT/RoBERTa**:
   - Captures bidirectional context and complex linguistic patterns.
   - Pretrained on large corpora, which can provide knowledge transfer.
   - Best performance typically requires fine-tuning on the target dataset.
   - Using pretrained models directly (without fine-tuning) may not always outperform simpler models.

### Factors Affecting Model Performance

1. **Text Length**:
   - Short texts may not provide enough context for deep learning models to leverage their strengths.
   - Traditional ML models might perform better on short texts with clear sentiment indicators.

2. **Domain Specificity**:
   - Performance depends on how well the model's training data matches the domain of the test data.
   - Pretrained models might not perform optimally on domain-specific language.

3. **Preprocessing**:
   - Different models benefit from different preprocessing approaches.
   - The preprocessing pipeline can significantly impact model performance.

4. **Hyperparameter Tuning**:
   - All models, especially deep learning models, benefit from proper hyperparameter tuning.
   - Default parameters might not be optimal for specific datasets.

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. Create a virtual environment:
```bash
python -m venv sentiment_env
source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Experiment

```bash
python main.py
```

This will train and evaluate all models on all datasets, saving the results in the `results` directory.

### Running Cross-Validation

```bash
python cross_validation.py
```

This will perform k-fold cross-validation for all models on all datasets, providing more robust performance metrics.

### Running Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

This will perform grid search to find the optimal hyperparameters for each model on each dataset.

## Results

The results are saved in the following formats:

- **CSV files**: Detailed metrics for each model and dataset
- **PNG images**: Visualizations of model performance, including:
  - Confusion matrices
  - Accuracy comparisons
  - Training time comparisons

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+
- scikit-learn 0.24+
- pandas, numpy, matplotlib, seaborn
- tqdm for progress tracking

## GPU Acceleration

The project automatically detects and uses available GPU resources. For optimal performance with deep learning models (BERT, RoBERTa, LSTM), a CUDA-compatible GPU is recommended.

## Customization

### Adding New Models

To add a new model:

1. Create a new file in the `models` directory
2. Implement a class with `train`, `predict`, and `evaluate` methods
3. Add the model to the model list in `main.py`, `cross_validation.py`, and `hyperparameter_tuning.py`

### Adding New Datasets

To add a new dataset:

1. Place the dataset file in the `data` directory
2. Add the dataset path to the `datasets` dictionary in the main scripts

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
