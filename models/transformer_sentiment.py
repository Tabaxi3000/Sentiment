import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import numpy as np
import joblib

class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx])
            
        return item

class SentimentTransformer(nn.Module):
    def __init__(self, dropout=0.1):
        super(SentimentTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)  # 768 is BERT's hidden size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take [CLS] token output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.sigmoid(linear_output)

class SentimentTransformerModel:
    def __init__(self, batch_size=16, epochs=3, learning_rate=2e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentimentTransformer().to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def train(self, X_train, y_train):
        """Train the model"""
        # Create dataset and dataloader
        train_dataset = SentimentDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}')
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return classification report"""
        test_dataset = SentimentDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].numpy()
                
                outputs = self.model(input_ids, attention_mask)
                predictions.extend((outputs.squeeze().cpu().numpy() > 0.5).astype(int))
                actual_labels.extend(labels)
        
        return classification_report(actual_labels, predictions)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        # Create dataset for single text
        dataset = SentimentDataset([text])
        
        self.model.eval()
        with torch.no_grad():
            input_ids = dataset[0]['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = dataset[0]['attention_mask'].unsqueeze(0).to(self.device)
            
            output = self.model(input_ids, attention_mask)
            pred = (output.squeeze().cpu().numpy() > 0.5).astype(int)
            
        return "Positive" if pred == 1 else "Negative", pred
    
    def save_model(self, filepath):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load the model from disk"""
        checkpoint = torch.load(filepath)
        model = cls(
            batch_size=checkpoint['model_config']['batch_size'],
            epochs=checkpoint['model_config']['epochs'],
            learning_rate=checkpoint['model_config']['learning_rate']
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model 