from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

class SentimentLogisticRegression:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return classification report"""
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        pred = self.model.predict([text])[0]
        return "Positive" if pred == 1 else "Negative", pred
    
    def save_model(self, filepath):
        """Save the model to disk"""
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load the model from disk"""
        return joblib.load(filepath) 