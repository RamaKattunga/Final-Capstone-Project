"""
Text Processing Module for Document Classification
Handles text cleaning, preprocessing, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK download failed. Some features may not work.")

class TextProcessor:
    def __init__(self):
        """Initialize the text processor"""
        try:
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            # Fallback if NLTK data is not available
            self.stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        self.vectorizer = None
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # TODO: Implement text cleaning steps
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # Text cleaning is like preparing ingredients before cooking:
        # 1. CONVERT TO LOWERCASE: "Hello World" becomes "hello world"
        #    - Why? So "Machine" and "machine" are treated as the same word
        #    - How? Use text.lower()
        # 2. REMOVE SPECIAL CHARACTERS: "AI & ML!" becomes "AI ML"
        #    - Why? Punctuation doesn't help classify documents
        #    - How? Use re.sub() to replace unwanted characters with spaces
        # 3. CLEAN UP EXTRA SPACES: "AI  ML" becomes "AI ML"
        #    - Why? Multiple spaces can confuse the algorithm
        #    - How? Use re.sub() to replace multiple spaces with single spaces
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Step 3: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove common stopwords from text
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text with stopwords removed
        """
        if not text:
            return ""
        
        # TODO: Remove stopwords from the text
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # Stopwords are common words like "the", "and", "is" that don't help classify documents
        # Think of it like removing filler words from a sentence to focus on important words
        # 
        # EXAMPLE: "The machine learning algorithm is very effective"
        # BECOMES: "machine learning algorithm effective"
        # 
        # PROCESS:
        # 1. Split the text into individual words: text.split()
        # 2. Check each word: is it a stopword? is it long enough to be useful?
        # 3. Keep only the good words: not in self.stop_words and len(word) > 2
        # 4. Join the good words back together: ' '.join()
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess_texts(self, texts):
        """
        Preprocess a list of texts
        
        Args:
            texts (list or pandas.Series): List of texts to preprocess
            
        Returns:
            list: Preprocessed texts
        """
        print("Preprocessing texts...")
        processed_texts = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:  # Progress indicator
                print(f"  Processing text {i+1}/{len(texts)}")
            
            # TODO: Apply cleaning and stopword removal
            # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
            # We need to clean each text in our dataset one by one
            # This is like washing each dish in a sink - we do them one at a time
            # 
            # PROCESS:
            # 1. Take the original text (might have uppercase, punctuation, etc.)
            # 2. Clean it using our clean_text method (lowercase, no punctuation)
            # 3. Remove stopwords using our remove_stopwords method (only important words)
            # 4. Store the processed text in our list
            # 
            # EXAMPLE: "The AI Technology is Amazing!" 
            # -> clean_text: "the ai technology is amazing"
            # -> remove_stopwords: "ai technology amazing"
            
            cleaned = self.clean_text(text)
            processed = self.remove_stopwords(cleaned)
            processed_texts.append(processed)
        
        print("Text preprocessing completed!")
        return processed_texts
    
    def create_features(self, train_texts, test_texts=None, max_features=5000):
        """
        Convert texts to numerical features using TF-IDF
        
        Args:
            train_texts (list): Training texts
            test_texts (list): Testing texts (optional)
            max_features (int): Maximum number of features to create
            
        Returns:
            tuple: (train_features, test_features) or just train_features
        """
        print(f"Creating TF-IDF features (max_features={max_features})...")
        
        # TODO: Create TF-IDF vectorizer and transform texts
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # TF-IDF converts text into numbers that computers can understand
        # Think of it like translating text into a special code for machines
        # 
        # WHAT IS TF-IDF?
        # - TF = Term Frequency: How often does a word appear in a document?
        # - IDF = Inverse Document Frequency: Is this word rare or common across all documents?
        # - Together: Words that appear often in one document but rarely in others are important
        # 
        # PROCESS:
        # 1. Create a TfidfVectorizer (the translator machine)
        # 2. Train it on our training texts (fit)
        # 3. Convert texts to numbers (transform)
        # 
        # EXAMPLE: "machine learning" might become [0.0, 0.5, 0.8, 0.0, ...]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # Additional stopword removal
            lowercase=True,
            ngram_range=(1, 2),  # Use single words and word pairs
            min_df=2,  # Ignore words that appear in less than 2 documents
            max_df=0.95  # Ignore words that appear in more than 95% of documents
        )
        
        # Fit and transform training data
        train_features = self.vectorizer.fit_transform(train_texts)
        print(f"Training features shape: {train_features.shape}")
        
        if test_texts is not None:
            # Transform test data (don't fit again)
            test_features = self.vectorizer.transform(test_texts)
            print(f"Test features shape: {test_features.shape}")
            return train_features, test_features
        
        return train_features
    
    def get_top_features(self, category_texts, n_features=10):
        """
        Get the most important features (words) for a category
        
        Args:
            category_texts (list): Texts from a specific category
            n_features (int): Number of top features to return
            
        Returns:
            list: Top features for the category
        """
        if not self.vectorizer:
            print("Please create features first using create_features()")
            return []
        
        # Combine all texts for the category
        combined_text = ' '.join(category_texts)
        
        # Get feature importances
        feature_vector = self.vectorizer.transform([combined_text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features
        feature_scores = feature_vector.toarray()[0]
        top_indices = np.argsort(feature_scores)[-n_features:][::-1]
        
        top_features = [feature_names[i] for i in top_indices if feature_scores[i] > 0]
        return top_features
    
    def analyze_vocabulary(self, texts, labels):
        """
        Analyze vocabulary differences between categories
        
        Args:
            texts (list): All texts
            labels (list): Corresponding category labels
            
        Returns:
            dict: Vocabulary analysis results
        """
        print("Analyzing vocabulary by category...")
        
        # Group texts by category
        category_texts = {}
        for text, label in zip(texts, labels):
            if label not in category_texts:
                category_texts[label] = []
            category_texts[label].append(text)
        
        # Get top features for each category
        vocab_analysis = {}
        for category, texts in category_texts.items():
            top_features = self.get_top_features(texts, n_features=15)
            vocab_analysis[category] = top_features
            print(f"\nTop words in {category}:")
            print(f"  {', '.join(top_features[:10])}")
        
        return vocab_analysis

# Example usage and testing
if __name__ == "__main__":
    print("Testing TextProcessor class...")
    
    # Sample texts for testing
    sample_texts = [
        "Machine learning algorithms are used for artificial intelligence applications.",
        "Clinical trials show promising results for new cancer treatment approaches.",
        "Financial markets demonstrate volatility due to economic uncertainty factors.",
        "Educational technology improves student learning outcomes significantly.",
        "Environmental sustainability requires immediate action on climate change issues."
    ]
    
    sample_labels = ['Technology', 'Healthcare', 'Finance', 'Education', 'Environment']
    
    # Create processor
    processor = TextProcessor()
    
    # Test text cleaning
    print("\nTesting text cleaning:")
    for i, text in enumerate(sample_texts[:2]):
        cleaned = processor.clean_text(text)
        no_stopwords = processor.remove_stopwords(cleaned)
        print(f"Original: {text}")
        print(f"Cleaned: {cleaned}")
        print(f"No stopwords: {no_stopwords}\n")
    
    # Test preprocessing
    processed_texts = processor.preprocess_texts(sample_texts)
    
    # Test feature creation
    train_features = processor.create_features(processed_texts)
    print(f"\nFeature matrix shape: {train_features.shape}")
    
    # Test vocabulary analysis
    vocab_analysis = processor.analyze_vocabulary(processed_texts, sample_labels)
