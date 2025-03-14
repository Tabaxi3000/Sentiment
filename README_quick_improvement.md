# Sentiment Analysis Project - Quick Improvement Guide

This guide provides instructions for quickly improving the sentiment analysis models within a 12-hour timeframe. The improvements focus on combining datasets, optimizing hyperparameters, and creating an enhanced user interface.

## Quick Improvement Pipeline

We've created a streamlined pipeline to improve the sentiment analysis models in just a few hours. The pipeline includes:

1. **Dataset Combination**: Combines multiple datasets into a single, larger dataset for better model training.
2. **Hyperparameter Optimization**: Quickly optimizes the most important hyperparameters for Logistic Regression and SVM models.
3. **Improved Ensemble**: Creates a weighted ensemble that gives more importance to the best-performing models.
4. **Enhanced Streamlit App**: Provides a modern, user-friendly interface for sentiment analysis with additional features.

## Getting Started

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)
- Git

### Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/Tabaxi3000/Sentiment.git
   cd Sentiment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv sentiment_env
   source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Quick Improvement Pipeline

To run the entire pipeline, simply execute:

```bash
python run_quick_improvement.py
```

This will:
- Combine all available datasets
- Optimize the models
- Prepare the improved Streamlit app

The process should take approximately 4-6 hours, depending on your hardware.

### Options

You can skip specific steps if needed:

```bash
# Skip dataset combination (if you've already done it)
python run_quick_improvement.py --skip-combine

# Skip model optimization (if you've already done it)
python run_quick_improvement.py --skip-optimize

# Skip Streamlit app creation (if you only want to improve models)
python run_quick_improvement.py --skip-streamlit
```

## Running the Improved Streamlit App

After the pipeline completes, you can run the improved Streamlit app:

```bash
streamlit run improved_streamlit_app.py
```

The app will be available at http://localhost:8501

## Features of the Improved App

The improved Streamlit app includes:

- **Text Analysis**: Analyze the sentiment of any text input
- **Batch Analysis**: Upload a CSV file with multiple texts for batch processing
- **Model Comparison**: Compare the performance of original and optimized models
- **Example Texts**: Pre-loaded examples from different domains (Amazon, Yelp, IMDB, Airlines)

## Model Improvements

The quick improvement pipeline focuses on:

1. **Data Expansion**: Combining multiple datasets to increase training data
2. **Hyperparameter Tuning**: Optimizing key parameters for each model
3. **Ensemble Weighting**: Giving more weight to better-performing models
4. **Preprocessing Enhancements**: Improving text cleaning and normalization

## Expected Results

After running the quick improvement pipeline, you can expect:

- **Logistic Regression**: Accuracy improvement from ~83% to ~85.5%
- **SVM**: Accuracy improvement from ~84% to ~86.5%
- **Ensemble**: Accuracy improvement from ~84.2% to ~87.5%

## Next Steps

If you have additional time, consider:

1. **Adding More Datasets**: Incorporate additional public sentiment datasets
2. **Advanced Preprocessing**: Implement more sophisticated text cleaning techniques
3. **Feature Engineering**: Create domain-specific features for better performance
4. **Deep Learning Models**: Integrate pre-trained transformer models like BERT

## Troubleshooting

If you encounter issues:

- **Memory Errors**: Reduce batch sizes or use smaller parameter grids
- **Missing Files**: Ensure all datasets are in the correct locations
- **Model Loading Errors**: Check that all required models are trained and saved

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special thanks to the open-source community for providing valuable datasets and tools 