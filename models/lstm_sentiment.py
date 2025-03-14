import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter
import re
import os
import joblib

class LSTMSentimentDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Create vocabulary if not provided
        if vocab is None:
            self.vocab = self._create_vocabulary(texts)
        else:
            self.vocab = vocab
            
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
    def _create_vocabulary(self, texts, max_vocab_size=10000):
        """Create vocabulary from texts"""
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # Select most common words
        vocab = ['<PAD>', '<UNK>']  # Special tokens
        vocab.extend([word for word, _ in word_counts.most_common(max_vocab_size - len(vocab))])
        return vocab
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on whitespace and removing punctuation"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()
    
    def _convert_to_indices(self, text):
        """Convert text to sequence of word indices"""
        words = self._tokenize(text)
        indices = [self.word_to_idx.get(word, 1) for word in words[:self.max_length]]  # 1 is <UNK>
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is <PAD>
            
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        indices = self._convert_to_indices(text)
        
        item = {
            'indices': torch.tensor(indices),
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx])
            
        return item

class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMSentimentModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # * 2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim * 2)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # (batch_size, hidden_dim * 2)
        
        # Apply dropout and pass through linear layer
        out = self.dropout(hidden)
        out = self.fc(out)
        
        return self.sigmoid(out)

class SentimentLSTM:
    def __init__(self, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.2, 
                 batch_size=64, epochs=5, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.vocab = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        # Create dataset and dataloader
        train_dataset = LSTMSentimentDataset(X_train, y_train)
        self.vocab = train_dataset.vocab
        
        # Initialize model
        self.model = LSTMSentimentModel(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                # Move batch to device
                indices = batch['indices'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(indices)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}')
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return classification report"""
        test_dataset = LSTMSentimentDataset(X_test, y_test, vocab=self.vocab)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                indices = batch['indices'].to(self.device)
                labels = batch['label'].numpy()
                
                outputs = self.model(indices)
                predictions.extend((outputs.squeeze().cpu().numpy() > 0.5).astype(int))
                actual_labels.extend(labels)
        
        return classification_report(actual_labels, predictions)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        # Create dataset for single text
        dataset = LSTMSentimentDataset([text], vocab=self.vocab)
        
        self.model.eval()
        with torch.no_grad():
            indices = dataset[0]['indices'].unsqueeze(0).to(self.device)
            
            output = self.model(indices)
            pred = (output.squeeze().cpu().numpy() > 0.5).astype(int)
            
        return "Positive" if pred == 1 else "Negative", pred
    
    def save_model(self, filepath):
        """Save the model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load the model from disk"""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model instance with saved configuration
        model = cls(
            embedding_dim=checkpoint['model_config']['embedding_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            num_layers=checkpoint['model_config']['num_layers'],
            dropout=checkpoint['model_config']['dropout'],
            batch_size=checkpoint['model_config']['batch_size'],
            epochs=checkpoint['model_config']['epochs'],
            learning_rate=checkpoint['model_config']['learning_rate']
        )
        
        # Set vocabulary
        model.vocab = checkpoint['vocab']
        
        # Initialize and load model state
        model.model = LSTMSentimentModel(
            vocab_size=len(model.vocab),
            embedding_dim=model.embedding_dim,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(model.device)
        
        return model 