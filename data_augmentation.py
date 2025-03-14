import pandas as pd
import numpy as np
import os
import random
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

def load_all_datasets():
    """Load and combine all datasets"""
    print("Loading all datasets...")
    datasets = {}
    
    # Load airlines dataset
    if os.path.exists('data/airlines_data.csv'):
        airlines_df = pd.read_csv('data/airlines_data.csv')
        datasets['airlines'] = airlines_df[['text', 'score']]
        print(f"Loaded airlines dataset: {len(datasets['airlines'])} samples")
    
    # Load Amazon dataset
    if os.path.exists('data/amazon_cells_labelled.txt'):
        amazon_df = pd.read_csv('data/amazon_cells_labelled.txt', sep='\t', header=None, names=['text', 'score'])
        datasets['amazon'] = amazon_df
        print(f"Loaded Amazon dataset: {len(datasets['amazon'])} samples")
    
    # Load Yelp dataset
    if os.path.exists('data/yelp_labelled.txt'):
        yelp_df = pd.read_csv('data/yelp_labelled.txt', sep='\t', header=None, names=['text', 'score'])
        datasets['yelp'] = yelp_df
        print(f"Loaded Yelp dataset: {len(datasets['yelp'])} samples")
    
    # Load IMDB dataset if available
    if os.path.exists('data/imdb_labelled.txt'):
        imdb_df = pd.read_csv('data/imdb_labelled.txt', sep='\t', header=None, names=['text', 'score'])
        datasets['imdb'] = imdb_df
        print(f"Loaded IMDB dataset: {len(datasets['imdb'])} samples")
    
    # Combine all datasets
    combined_df = pd.concat([df for df in datasets.values()], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")
    
    return combined_df, datasets

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and len(synonym) > 3:  # Avoid short words
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms"""
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if len(word) > 3]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def random_insertion(text, n=1):
    """Randomly insert n words into the text"""
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
    
    return ' '.join(new_words)

def add_word(words):
    """Add a random synonym of a random word to a random position"""
    if len(words) == 0:
        return
    
    random_word = random.choice([word for word in words if len(word) > 3])
    synonyms = get_synonyms(random_word)
    
    if len(synonyms) > 0:
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, random_synonym)

def random_swap(text, n=1):
    """Randomly swap n pairs of words in the text"""
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    
    for _ in range(n):
        if len(new_words) > 1:  # Need at least 2 words to swap
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    """Randomly delete words from the text with probability p"""
    words = nltk.word_tokenize(text)
    
    # Don't delete too much if the text is already short
    if len(words) <= 5:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    # Make sure we don't delete all words
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)

def load_bert_for_augmentation():
    """Load BERT model for masked language model augmentation"""
    print("Loading BERT model for text augmentation...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return tokenizer, model, device

def bert_augmentation(text, tokenizer, model, device, mask_prob=0.15):
    """Use BERT masked language model to replace words"""
    tokens = tokenizer.tokenize(text)
    
    # Don't augment if text is too short
    if len(tokens) <= 5:
        return text
    
    # Create a copy of tokens and randomly mask some
    masked_tokens = tokens.copy()
    mask_indices = []
    
    for i in range(len(tokens)):
        if random.random() < mask_prob and tokens[i] not in ['[CLS]', '[SEP]', '.', ',', '!', '?']:
            masked_tokens[i] = '[MASK]'
            mask_indices.append(i)
    
    # If no tokens were masked, mask one random token
    if len(mask_indices) == 0:
        valid_indices = [i for i, token in enumerate(tokens) 
                         if token not in ['[CLS]', '[SEP]', '.', ',', '!', '?']]
        if valid_indices:
            idx = random.choice(valid_indices)
            masked_tokens[idx] = '[MASK]'
            mask_indices.append(idx)
    
    # Convert to input IDs and create tensors
    inputs = tokenizer.encode(masked_tokens, return_tensors='pt').to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.logits
    
    # Replace masked tokens with predictions
    for idx in mask_indices:
        # Get the predicted token ID for this position
        predicted_token_id = predictions[0, idx+1].argmax().item()  # +1 for [CLS] token
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
        
        # Replace in the original tokens list
        if predicted_token != tokens[idx]:  # Only replace if different
            tokens[idx] = predicted_token
    
    # Convert back to text
    augmented_text = tokenizer.convert_tokens_to_string(tokens)
    
    return augmented_text

def augment_dataset(df, augmentation_factor=2, use_bert=True):
    """Augment the dataset using various techniques"""
    print(f"Augmenting dataset with factor {augmentation_factor}...")
    original_size = len(df)
    augmented_texts = []
    augmented_labels = []
    
    # Load BERT model if needed
    if use_bert:
        tokenizer, model, device = load_bert_for_augmentation()
    
    # Create progress bar
    pbar = tqdm(total=original_size * (augmentation_factor - 1), desc="Augmenting data")
    
    # For each sample in the dataset
    for idx, row in df.iterrows():
        text = row['text']
        label = row['score']
        
        # Number of augmentations to create for this sample
        num_aug = augmentation_factor - 1
        
        for _ in range(num_aug):
            # Randomly choose augmentation technique
            aug_type = random.choice(['synonym', 'insert', 'swap', 'delete', 'bert'])
            
            if aug_type == 'synonym':
                aug_text = synonym_replacement(text, n=random.randint(1, 3))
            elif aug_type == 'insert':
                aug_text = random_insertion(text, n=random.randint(1, 2))
            elif aug_type == 'swap':
                aug_text = random_swap(text, n=random.randint(1, 2))
            elif aug_type == 'delete':
                aug_text = random_deletion(text, p=0.1)
            elif aug_type == 'bert' and use_bert:
                aug_text = bert_augmentation(text, tokenizer, model, device)
            else:
                aug_text = synonym_replacement(text, n=1)
            
            # Add augmented sample
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    
    # Create augmented dataframe
    aug_df = pd.DataFrame({
        'text': augmented_texts,
        'score': augmented_labels
    })
    
    # Combine with original data
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f"Original dataset size: {original_size}")
    print(f"Augmented dataset size: {len(combined_df)}")
    
    return combined_df

def save_augmented_dataset(df, output_path='data/augmented_data.csv'):
    """Save the augmented dataset to a CSV file"""
    print(f"Saving augmented dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load and combine all datasets
    combined_df, individual_datasets = load_all_datasets()
    
    # Augment the combined dataset
    augmented_df = augment_dataset(combined_df, augmentation_factor=2, use_bert=True)
    
    # Save the augmented dataset
    save_augmented_dataset(augmented_df, 'data/augmented_data.csv') 