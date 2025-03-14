# Sentiment Analysis Streamlit App

This Streamlit app provides a user-friendly web interface for sentiment analysis using machine learning models trained on a combined dataset of Amazon reviews, Yelp reviews, IMDB movie reviews, and airline tweets.

## Features

- **Single Text Analysis**: Analyze the sentiment of a single text input.
- **Batch Analysis**: Process multiple texts at once by uploading a file or entering multiple lines.
- **Example Texts**: Try out the sentiment analysis with pre-defined examples.
- **Multiple Models**: Choose from different models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - AdaBoost
  - Equal Weights Ensemble
  - Weighted Ensemble (recommended)
- **Confidence Scores**: Get confidence scores for predictions when using ensemble models.
- **Individual Model Predictions**: See how each model in the ensemble voted.
- **Batch Statistics**: View statistics and visualizations for batch predictions.
- **Download Results**: Download batch analysis results as CSV.

## Installation

1. Make sure you have Python 3.7+ installed.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the trained models are available in the `models/combined/` directory.

## Running the App

To run the Streamlit app, use the following command:

```bash
streamlit run streamlit_app.py
```

This will start a local web server and open the app in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Usage

### Single Text Analysis

1. Enter your text in the text area.
2. Click "Analyze Sentiment".
3. View the sentiment prediction and confidence score.

### Batch Analysis

1. Choose between "Text Input" or "File Upload".
2. For text input, enter multiple texts (one per line).
3. For file upload, upload a TXT file (one text per line) or a CSV file (with a 'text' column).
4. Click "Analyze Batch".
5. View the results table, statistics, and visualizations.
6. Download the results as a CSV file if needed.

### Examples

1. Click on any example to analyze its sentiment.
2. View the sentiment prediction and confidence score.

## Models

- **Logistic Regression**: A linear model for binary classification. Accuracy: 83.09%
- **Support Vector Machine (SVM)**: A powerful classification algorithm. Accuracy: 84.00%
- **AdaBoost**: An ensemble method that combines multiple weak classifiers. Accuracy: 79.27%
- **Equal Weights Ensemble**: Combines predictions from all models with equal weights. Accuracy: 83.82%
- **Weighted Ensemble**: Combines predictions with different weights (SVM has higher weight). Accuracy: 84.18%

## Customization

You can customize the app by modifying the `streamlit_app.py` file:

- Add more models
- Change the UI layout
- Add more features
- Customize the styling

## Troubleshooting

If you encounter any issues:

1. Make sure all required packages are installed.
2. Check that the model files exist in the correct directory.
3. Verify that you're running the app from the project root directory.
4. Check the console for any error messages.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 