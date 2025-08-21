import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Use top 5000 most important words
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Use single words and word pairs
        )
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # TODO: Add your text cleaning code here
        # Hint: Remove special characters, convert to lowercase
        # Remove numbers, remove extra spaces
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        # TODO: Load the CSV file using pandas
        # Hint: Use pd.read_csv()
        
        try:
            data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print("Error: Dataset file not found!")
            return None
    
    def prepare_data(self, data):
        """
        Prepare data for machine learning
        
        Args:
            data (pandas.DataFrame): Raw dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Clean the text data
        print("Cleaning text data...")
        data['cleaned_text'] = data['text'].apply(self.clean_text)
        
        # Prepare features (X) and target (y)
        X = data['cleaned_text']
        y = data['category']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert text to numerical features
        print("Converting text to numerical features...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print(f"Training set size: {X_train_vectorized.shape}")
        print(f"Testing set size: {X_test_vectorized.shape}")
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load data
    data = processor.load_data('data/research_papers_dataset.csv')
    
    if data is not None:
        # Show basic information about the dataset
        print("\nDataset Info:")
        print(data.info())
        print("\nCategory distribution:")
        print(data['category'].value_counts())
