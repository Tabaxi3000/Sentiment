import pandas as pd
import os
from pathlib import Path

def load_data_set_1(data_dir='data'):
    """
    Load and process sentiment data files from the data directory
    
    Args:
        data_dir (str): Path to data directory, defaults to 'data'
        
    Returns:
        pandas.DataFrame: Combined and shuffled dataset with 'text' and 'score' columns
    """
    # Create Path object
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Directory '{data_dir}' not found")
    
    # Initialize list to store all data
    all_data = []
    
    # Process each text file
    for file in data_path.glob('*_labelled.txt'):
        try:
            # Read file content
            df = pd.read_csv(file, sep='\t', header=None, names=['text', 'score'])
            print(f"Successfully processed {file.name}")
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    # Combine all dataframes
    if not all_data:
        raise ValueError("No valid data files found")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the combined dataset
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return shuffled_df

def load_data_airlines(data_dir='data'):
    """
    Load and process airlines review data
    
    Args:
        data_dir (str): Path to data directory, defaults to 'data'
        
    Returns:
        pandas.DataFrame: Processed and shuffled dataset with 'text' and 'score' columns
    """
    # Create Path object
    data_path = Path(data_dir) / 'airlines_reviews.csv'
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"File '{data_path}' not found")
    
    try:
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Create new dataframe with required columns
        processed_df = pd.DataFrame({
            'text': df['Reviews'],
            'score': df['Recommended'].map({'yes': 1, 'no': 0})
        })
        
        # Shuffle the dataset
        shuffled_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("Successfully processed airlines reviews data")
        return shuffled_df
        
    except Exception as e:
        print(f"Error processing airlines data: {str(e)}")
        raise

def load_combined_data(data_dir='data'):
    """
    Load and combine both sentiment and airlines datasets
    
    Args:
        data_dir (str): Path to data directory, defaults to 'data'
        
    Returns:
        pandas.DataFrame: Combined and shuffled dataset with 'text' and 'score' columns
    """
    try:
        # Load both datasets
        sentiment_data = load_data_set_1(data_dir)
        airlines_data = load_data_airlines(data_dir)
        
        # Combine datasets
        combined_df = pd.concat([sentiment_data, airlines_data], ignore_index=True)
        
        # Shuffle the combined dataset
        shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\nSuccessfully combined both datasets")
        return shuffled_df
        
    except Exception as e:
        print(f"Error combining datasets: {str(e)}")
        raise

if __name__ == "__main__":
    airline_data = load_data_airlines()
    airline_data.to_csv('data/airlines_data.csv', index=False)

    # Load combined data
    combined_data = load_combined_data()
    
    # Save to CSV
    combined_data.to_csv('data/combined_data.csv', index=False)
    
    # Print summary
    print("\nCombined Data Summary:")
    print(f"Total samples: {len(combined_data)}")
    print(f"Positive samples: {sum(combined_data['score'] == 1)}")
    print(f"Negative samples: {sum(combined_data['score'] == 0)}")
    print("\nSample distribution:")
    print(f"Sentiment data samples: {len(load_data_set_1())}")
    print(f"Airlines data samples: {len(load_data_airlines())}")
    print("\nFirst few samples:")
    print(combined_data.head())
