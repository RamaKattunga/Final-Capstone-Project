"""
Data Loading Module for Document Classification
Handles loading and basic preprocessing of research papers dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path='data/research_papers_dataset.csv'):
        """
        Initialize the data loader
        
        Args:
            file_path (str): Path to the CSV dataset file
        """
        self.file_path = file_path
        self.data = None
        self.categories = None
        
    def load_dataset(self):
        """
        Load the research papers dataset from CSV file
        
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            # TODO: Load the CSV file using pandas
            # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
            # 1. We use pandas (pd) to read CSV files - it's like opening Excel but in Python
            # 2. pd.read_csv() is the function that reads CSV files
            # 3. self.file_path contains the location of our dataset file
            # 4. We store the loaded data in self.data so we can use it later
            # 
            # Think of this like: "Open the research papers spreadsheet and put it in memory"
            
            self.data = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.data.shape}")
            
            # Get unique categories
            self.categories = sorted(self.data['category'].unique())
            print(f"Categories found: {self.categories}")
            
            return self.data
            
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.file_path}")
            print("Please ensure the dataset file is in the correct location.")
            return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def explore_dataset(self):
        """
        Display basic information about the dataset
        """
        if self.data is None:
            print("Please load the dataset first using load_dataset()")
            return
        
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        # Basic information
        print(f"Total papers: {len(self.data)}")
        print(f"Number of categories: {len(self.categories)}")
        
        # Category distribution
        print("\nCategory Distribution:")
        category_counts = self.data['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {category}: {count} papers ({percentage:.1f}%)")
        
        # Text length statistics
        print("\nText Length Statistics:")
        self.data['text_length'] = self.data['text'].str.len()
        print(f"  Average length: {self.data['text_length'].mean():.0f} characters")
        print(f"  Shortest paper: {self.data['text_length'].min()} characters")
        print(f"  Longest paper: {self.data['text_length'].max()} characters")
        
        # Sample data
        print("\nSample Papers:")
        for i in range(2):
            print(f"\nPaper {i+1}:")
            print(f"  Title: {self.data.iloc[i]['title'][:80]}...")
            print(f"  Category: {self.data.iloc[i]['category']}")
            print(f"  Abstract preview: {self.data.iloc[i]['abstract'][:100]}...")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets
        
        Args:
            test_size (float): Proportion of dataset for testing (0.2 = 20%)
            random_state (int): Random seed for reproducible results
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            print("Please load the dataset first using load_dataset()")
            return None, None, None, None
        
        # TODO: Split the data into features (X) and labels (y)
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # 1. In machine learning, we need to separate our data into two parts:
        #    - FEATURES (X): The information we use to make predictions (the text of papers)
        #    - LABELS (y): What we want to predict (the category of each paper)
        # 2. Think of it like this: 
        #    - X = "What does the paper say?" (the content we analyze)
        #    - y = "What category should it be in?" (the answer we want to predict)
        # 3. self.data['text'] gets the 'text' column (contains title + abstract)
        # 4. self.data['category'] gets the 'category' column (Technology, Healthcare, etc.)
        
        X = self.data['text']  # Features: the text content
        y = self.data['category']  # Labels: the categories
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Ensures each category is proportionally represented
        )
        
        print(f"\nData split completed:")
        print(f"Training set: {len(X_train)} papers")
        print(f"Testing set: {len(X_test)} papers")
        
        # Show category distribution in splits
        print(f"\nTraining set distribution:")
        train_dist = y_train.value_counts()
        for category, count in train_dist.items():
            print(f"  {category}: {count} papers")
        
        return X_train, X_test, y_train, y_test
    
    def get_sample_data(self, n_samples=5):
        """
        Get a small sample of data for testing purposes
        
        Args:
            n_samples (int): Number of samples to return
            
        Returns:
            pandas.DataFrame: Sample data
        """
        if self.data is None:
            print("Please load the dataset first using load_dataset()")
            return None
        
        return self.data.head(n_samples)

# Example usage and testing
if __name__ == "__main__":
    print("Testing DataLoader class...")
    
    # Create data loader
    loader = DataLoader()
    
    # Load and explore dataset
    dataset = loader.load_dataset()
    
    if dataset is not None:
        loader.explore_dataset()
        
        # Test data splitting
        X_train, X_test, y_train, y_test = loader.split_data()
        
        # Show sample data
        print("\nSample data:")
        sample = loader.get_sample_data(3)
        if sample is not None:
            for idx, row in sample.iterrows():
                print(f"\nSample {idx + 1}:")
                print(f"  Category: {row['category']}")
                print(f"  Title: {row['title'][:60]}...")
    else:
        print("Failed to load dataset. Please check file location.")
