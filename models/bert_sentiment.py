import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import numpy as np
import os
import joblib
from tqdm import tqdm

class BertSentimentDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=128, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False, use_auth_token=False)
        else:
            self.tokenizer = tokenizer
    
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
        
        # Add token_type_ids only if present in the encoding
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx])
            
        return item

class SentimentBERT:
    def __init__(self, batch_size=16, epochs=3, learning_rate=2e-5, use_pretrained=True, pretrained_model="distilbert-base-uncased-finetuned-sst-2-english"):
        # Check for GPU availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set CUDA options for better performance if GPU is available
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("CUDA optimizations enabled")
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_pretrained = use_pretrained
        self.pretrained_model = pretrained_model
        
        if use_pretrained:
            try:
                print(f"Loading pretrained model: {pretrained_model}")
                # Use AutoTokenizer and AutoModelForSequenceClassification for better compatibility
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=False, use_auth_token=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, local_files_only=False, use_auth_token=False)
                self.model.to(self.device)
                print("Pretrained model loaded successfully")
                
                # Check if the model has the expected number of labels
                num_labels = self.model.config.num_labels
                print(f"Model has {num_labels} output labels")
                
                # Store the label mapping (for binary sentiment)
                if num_labels == 2:
                    self.label_mapping = {0: 0, 1: 1}  # Direct mapping
                elif num_labels == 3:
                    # For models with 3 labels (negative, neutral, positive)
                    # Map 0 (negative) to 0, and 1,2 (neutral, positive) to 1
                    self.label_mapping = {0: 0, 1: 1, 2: 1}
                else:
                    # Default mapping - assume 0 is negative, highest is positive
                    self.label_mapping = {0: 0, num_labels-1: 1}
                    
                print(f"Using label mapping: {self.label_mapping}")
                
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Falling back to custom BERT model")
                self.use_pretrained = False
        
        if not self.use_pretrained:
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False, use_auth_token=False)
                self.model = BertSentimentClassifier().to(self.device)
                print("Custom BERT model loaded successfully")
            except Exception as e:
                print(f"Error initializing BERT model: {e}")
                raise
        
    def train(self, X_train, y_train):
        """Train the model"""
        # If using pretrained model, skip training
        if self.use_pretrained:
            print("Using pretrained model - skipping training")
            return
            
        # Create dataset and dataloader
        train_dataset = BertSentimentDataset(X_train, y_train, tokenizer=self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=4 if self.device.type == 'cuda' else 0
        )
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        print(f"Training BERT model on {len(X_train)} samples...")
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            total_loss = 0
            
            # Create progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Check if token_type_ids is in the batch
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Update total loss and progress bar
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Average Loss: {avg_loss:.4f}')
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return classification report"""
        if self.use_pretrained:
            # For pretrained model, we'll evaluate directly
            y_pred = []
            
            # Create progress bar for evaluation
            progress_bar = tqdm(range(len(X_test)), desc="Evaluating", leave=True)
            
            for i in progress_bar:
                text = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
                sentiment, score = self.predict(text)
                y_pred.append(score)
                
                # Update progress bar with current accuracy
                if i > 0:
                    current_acc = np.mean(np.array(y_pred) == np.array(y_test[:i+1]))
                    progress_bar.set_postfix({'accuracy': f"{current_acc:.4f}"})
            
            return classification_report(y_test, y_pred)
        else:
            # For custom model, use the original evaluation method
            test_dataset = BertSentimentDataset(X_test, y_test, tokenizer=self.tokenizer)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size,
                pin_memory=True if self.device.type == 'cuda' else False,
                num_workers=4 if self.device.type == 'cuda' else 0
            )
            
            self.model.eval()
            predictions = []
            actual_labels = []
            
            # Create progress bar for evaluation
            progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
            
            with torch.no_grad():
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].numpy()
                    
                    # Check if token_type_ids is in the batch
                    if 'token_type_ids' in batch:
                        token_type_ids = batch['token_type_ids'].to(self.device)
                        outputs = self.model(input_ids, attention_mask, token_type_ids)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                        
                    batch_preds = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
                    
                    predictions.extend(batch_preds)
                    actual_labels.extend(labels)
                    
                    # Update progress bar with current accuracy
                    current_acc = np.mean(np.array(batch_preds) == np.array(labels))
                    progress_bar.set_postfix({'batch_acc': f"{current_acc:.4f}"})
            
            return classification_report(actual_labels, predictions)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if self.use_pretrained:
            # For pretrained model
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                
                # Map the model's output to binary sentiment
                # For models with 2 classes, this is straightforward
                # For models with 3 classes (negative, neutral, positive), we need to map
                binary_pred = self.label_mapping.get(prediction, 1)  # Default to positive if unknown
                
            return "Positive" if binary_pred == 1 else "Negative", binary_pred
        else:
            # For custom model
            # Create dataset for single text
            dataset = BertSentimentDataset([text], tokenizer=self.tokenizer)
            
            self.model.eval()
            with torch.no_grad():
                input_ids = dataset[0]['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = dataset[0]['attention_mask'].unsqueeze(0).to(self.device)
                
                # Check if token_type_ids is in the dataset item
                if 'token_type_ids' in dataset[0]:
                    token_type_ids = dataset[0]['token_type_ids'].unsqueeze(0).to(self.device)
                    output = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    output = self.model(input_ids, attention_mask)
                    
                pred = (output.squeeze().cpu().numpy() > 0.5).astype(int)
                
            return "Positive" if pred == 1 else "Negative", pred
    
    def save_model(self, filepath):
        """Save the model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model configuration and state
        model_data = {
            'use_pretrained': self.use_pretrained,
            'pretrained_model': self.pretrained_model,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }
        
        if not self.use_pretrained:
            model_data['model_state_dict'] = self.model.state_dict()
            
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load the model from disk"""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model instance
        model = cls(
            batch_size=checkpoint.get('batch_size', 16),
            epochs=checkpoint.get('epochs', 3),
            learning_rate=checkpoint.get('learning_rate', 2e-5),
            use_pretrained=checkpoint.get('use_pretrained', False),
            pretrained_model=checkpoint.get('pretrained_model', "distilbert-base-uncased-finetuned-sst-2-english")
        )
        
        # Load state dict for custom model
        if not checkpoint.get('use_pretrained', False) and 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
            
        return model


class BertSentimentClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertSentimentClassifier, self).__init__()
        try:
            # Try to load the model with default parameters
            self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=False, use_auth_token=False)
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Trying alternative loading method...")
            # Try with different parameters
            self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=False, use_auth_token=False, 
                                                 mirror='https://huggingface.co', force_download=True)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)  # 768 is BERT's hidden size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Check if token_type_ids is provided
        if token_type_ids is not None:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        pooled_output = outputs.pooler_output  # [CLS] token output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits) 