from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

class SentimentAdaBoost:
    def __init__(self, n_estimators=100, learning_rate=1.0):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42))
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