import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from tqdm import tqdm

class SentimentRoBERTa:
    def __init__(self, model_name='roberta-base', epochs=3, batch_size=16, max_length=128, use_pretrained=True, pretrained_model="cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Initialize the RoBERTa model for sentiment analysis
        
        Args:
            model_name (str): Name of the pre-trained RoBERTa model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length
            use_pretrained (bool): Whether to use a pretrained sentiment model
            pretrained_model (str): Name of the pretrained sentiment model
        """
        # Check for GPU availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set CUDA options for better performance if GPU is available
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("CUDA optimizations enabled")
            
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_pretrained = use_pretrained
        self.pretrained_model = pretrained_model
        
        if use_pretrained:
            try:
                print(f"Loading pretrained sentiment model: {pretrained_model}")
                # Use AutoTokenizer and AutoModelForSequenceClassification for better compatibility
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=False, use_auth_token=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, local_files_only=False, use_auth_token=False)
                self.model.to(self.device)
                print("Pretrained RoBERTa sentiment model loaded successfully")
                
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
                print(f"Error loading pretrained RoBERTa model: {e}")
                print("Falling back to custom RoBERTa model")
                self.use_pretrained = False
        
        if not self.use_pretrained:
            # Load tokenizer and model
            try:
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=False, use_auth_token=False)
                print("RoBERTa tokenizer loaded successfully")
            except Exception as e:
                print(f"Error loading RoBERTa tokenizer: {e}")
                print("Trying alternative loading method...")
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=False, use_auth_token=False, 
                                                                mirror='https://huggingface.co', force_download=True)
            
            try:
                self.model = RoBERTaClassifier(model_name)
                self.model.to(self.device)
                print("Custom RoBERTa model loaded successfully")
            except Exception as e:
                print(f"Error initializing RoBERTa model: {e}")
                raise
        
    def train(self, X_train, y_train):
        """
        Train the RoBERTa model
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        # If using pretrained model, skip training
        if self.use_pretrained:
            print("Using pretrained model - skipping training")
            return
            
        # Prepare dataset
        train_dataset = self._prepare_dataset(X_train, y_train)
        
        # Create data loader with GPU optimizations
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=4 if self.device.type == 'cuda' else 0
        )
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"Training RoBERTa model on {len(X_train)} samples...")
        
        # Track best loss for early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 2  # Number of epochs with no improvement before early stopping
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Set model to training mode
            self.model.train()
            
            # Track loss
            total_loss = 0
            
            # Progress bar with more information
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.epochs}", 
                leave=True,
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Clear gradients
                optimizer.zero_grad()
                
                # Get inputs
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update progress bar with more information
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                print(f"New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the RoBERTa model
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            str: Classification report
        """
        if self.use_pretrained:
            # For pretrained model, evaluate directly
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
            # Prepare dataset
            test_dataset = self._prepare_dataset(X_test, y_test)
            
            # Create data loader with GPU optimizations
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True if self.device.type == 'cuda' else False,
                num_workers=4 if self.device.type == 'cuda' else 0
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Track predictions and true labels
            all_preds = []
            all_labels = []
            
            # Evaluate without gradient calculation
            with torch.no_grad():
                # Enhanced progress bar for evaluation
                progress_bar = tqdm(
                    test_dataloader, 
                    desc="Evaluating", 
                    leave=True,
                    unit="batch",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
                )
                
                for batch in progress_bar:
                    # Get inputs
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get predictions
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    batch_labels = labels.cpu().numpy()
                    
                    # Add to lists
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels)
                    
                    # Update progress bar with current accuracy
                    batch_acc = np.mean(preds == batch_labels)
                    progress_bar.set_postfix({'batch_acc': f"{batch_acc:.4f}"})
            
            # Calculate overall accuracy
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            print(f"Overall accuracy: {accuracy:.4f}")
            
            # Generate classification report
            report = classification_report(all_labels, all_preds)
            
            return report
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment, score) where sentiment is "Positive" or "Negative"
                  and score is 1 for positive, 0 for negative
        """
        if self.use_pretrained:
            # For pretrained model
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                
                # Map the model's output to binary sentiment using the label mapping
                binary_pred = self.label_mapping.get(prediction, 1)  # Default to positive if unknown
                
            return "Positive" if binary_pred == 1 else "Negative", binary_pred
        else:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Predict without gradient calculation
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            
            # Return sentiment and score
            sentiment = "Positive" if pred == 1 else "Negative"
            
            return sentiment, pred
    
    def save_model(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state dict and config
        model_dict = {
            'use_pretrained': self.use_pretrained,
            'pretrained_model': self.pretrained_model,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        if not self.use_pretrained:
            model_dict['model_state_dict'] = self.model.state_dict()
        
        torch.save(model_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load the model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            SentimentRoBERTa: Loaded model
        """
        # Load model dict
        model_dict = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create instance
        instance = cls(
            model_name=model_dict.get('model_name', 'roberta-base'),
            max_length=model_dict.get('max_length', 128),
            batch_size=model_dict.get('batch_size', 16),
            epochs=model_dict.get('epochs', 3),
            use_pretrained=model_dict.get('use_pretrained', False),
            pretrained_model=model_dict.get('pretrained_model', "cardiffnlp/twitter-roberta-base-sentiment")
        )
        
        # Load state dict for custom model
        if not model_dict.get('use_pretrained', False) and 'model_state_dict' in model_dict:
            instance.model.load_state_dict(model_dict['model_state_dict'])
        
        return instance
    
    def _prepare_dataset(self, texts, labels):
        """
        Prepare dataset for training or evaluation
        
        Args:
            texts: Input texts
            labels: Input labels
            
        Returns:
            list: List of dictionaries with input_ids, attention_mask, and labels
        """
        dataset = []
        
        # Use tqdm for dataset preparation
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Preparing dataset"):
            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create dataset item
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
            dataset.append(item)
        
        return dataset


class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        """
        RoBERTa classifier for sentiment analysis
        
        Args:
            model_name (str): Name of the pre-trained RoBERTa model
            num_labels (int): Number of output labels
        """
        super(RoBERTaClassifier, self).__init__()
        
        # Load RoBERTa model
        try:
            self.roberta = RobertaModel.from_pretrained(model_name, local_files_only=False, use_auth_token=False)
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            print("Trying alternative loading method...")
            self.roberta = RobertaModel.from_pretrained(model_name, local_files_only=False, use_auth_token=False,
                                                      mirror='https://huggingface.co', force_download=True)
        
        # Classifier head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for loss calculation
            
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Model outputs
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled output (CLS token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Create output object
        class ModelOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return ModelOutput(loss, logits) 