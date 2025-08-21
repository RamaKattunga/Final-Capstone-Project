import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from collections import Counter
import html

class AdvancedDataPreprocessor:
    def __init__(self):
        """Initialize the advanced preprocessor"""
        self.label_encoder = LabelEncoder()
        self.category_mapping = {}
        self.vectorizer = None
        self.feature_names = None
        
    def clean_html_and_special_chars(self, text):
        """
        Remove HTML tags and decode special characters
        
        Args:
            text (str): Text with potential HTML and special characters
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # TODO: Implement HTML tag removal and special character handling
        # Hints: 
        # - Use html.unescape() for HTML entities
        # - Use re.sub() to remove HTML tags
        # - Handle common special characters
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def handle_missing_data(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df (pandas.DataFrame): Dataset with potential missing values
            
        Returns:
            pandas.DataFrame: Dataset with missing values handled
        """
        print("Handling missing data...")
        
        # TODO: Implement missing data handling strategy
        # Hints:
        # - For missing abstracts, you might use titles
        # - For missing titles, you might use part of abstracts
        # - Remove rows that are completely empty
        
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(subset=['title', 'abstract'], how='all')
        
        # Handle missing abstracts - use title as abstract
        missing_abstract = df_clean['abstract'].isna() | (df_clean['abstract'] == '')
        df_clean.loc[missing_abstract, 'abstract'] = df_clean.loc[missing_abstract, 'title']
        
        # Handle missing titles - use first part of abstract
        missing_title = df_clean['title'].isna() | (df_clean['title'] == '')
        df_clean.loc[missing_title, 'title'] = df_clean.loc[missing_title, 'abstract'].str[:50] + "..."
        
        # Remove rows where both title and abstract are still empty
        df_clean = df_clean[(df_clean['title'] != '') & (df_clean['abstract'] != '')]
        
        print(f"Removed {len(df) - len(df_clean)} rows with insufficient data")
        return df_clean
    
    def standardize_categories(self, df):
        """
        Standardize category labels to consistent format
        
        Args:
            df (pandas.DataFrame): Dataset with inconsistent categories
            
        Returns:
            pandas.DataFrame: Dataset with standardized categories
        """
        print("Standardizing categories...")
        
        # TODO: Create mapping for category variations
        # Hint: Map similar categories to standard names
        
        # Define category mapping
        category_mapping = {
            # Technology variations
            'tech': 'Technology',
            'technology': 'Technology', 
            'computer science': 'Technology',
            'cs': 'Technology',
            
            # Healthcare variations
            'health': 'Healthcare',
            'healthcare': 'Healthcare',
            'medical': 'Healthcare',
            'medicine': 'Healthcare',
            
            # Finance variations
            'financial': 'Finance',
            'finance': 'Finance',
            'economics': 'Finance',
            'economic': 'Finance',
            
            # Education variations
            'educational': 'Education',
            'education': 'Education',
            'learning': 'Education',
            'teaching': 'Education',
            
            # Environment variations
            'environmental': 'Environment',
            'environment': 'Environment',
            'climate': 'Environment',
            'green': 'Environment'
        }
        
        df_clean = df.copy()
        
        # Standardize categories
        df_clean['category'] = df_clean['category'].str.lower().str.strip()
        df_clean['category'] = df_clean['category'].map(category_mapping).fillna(df_clean['category'])
        
        # Capitalize properly
        df_clean['category'] = df_clean['category'].str.title()
        
        # Store the mapping for later use
        self.category_mapping = category_mapping
        
        print("Category distribution after standardization:")
        print(df_clean['category'].value_counts())
        
        return df_clean
    
    def remove_duplicates(self, df, similarity_threshold=0.8):
        """
        Remove duplicate entries based on text similarity
        
        Args:
            df (pandas.DataFrame): Dataset with potential duplicates
            similarity_threshold (float): Threshold for considering entries as duplicates
            
        Returns:
            pandas.DataFrame: Dataset with duplicates removed
        """
        print("Removing duplicates...")
        
        # TODO: Implement duplicate detection and removal
        # Hints:
        # - Use string similarity metrics
        # - Consider both exact and near-duplicates
        # - Be careful not to remove legitimate similar papers
        
        df_clean = df.copy()
        
        # Remove exact title duplicates
        before_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['title'], keep='first')
        exact_duplicates_removed = before_count - len(df_clean)
        
        # Remove near duplicates based on title similarity
        # Simple approach: remove titles that are very similar after basic cleaning
        df_clean['title_clean'] = df_clean['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df_clean = df_clean.drop_duplicates(subset=['title_clean'], keep='first')
        df_clean = df_clean.drop(columns=['title_clean'])
        
        near_duplicates_removed = before_count - exact_duplicates_removed - len(df_clean)
        
        print(f"Removed {exact_duplicates_removed} exact duplicates")
        print(f"Removed {near_duplicates_removed} near duplicates")
        print(f"Dataset size after deduplication: {len(df_clean)}")
        
        return df_clean
    
    def advanced_feature_engineering(self, df):
        """
        Create advanced features for better classification
        
        Args:
            df (pandas.DataFrame): Clean dataset
            
        Returns:
            tuple: (feature_matrix, labels, feature_names)
        """
        print("Creating advanced features...")
        
        # TODO: Implement advanced feature engineering
        # Hints:
        # - Use TF-IDF with different n-gram ranges
        # - Add text length features
        # - Add vocabulary richness features
        # - Combine multiple feature types
        
        # Clean text data
        df['text_clean'] = df.apply(lambda row: 
            self.clean_html_and_special_chars(row['title'] + ' ' + row['abstract']), axis=1)
        
        # Create TF-IDF features with multiple n-gram ranges
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            min_df=2,            # Ignore terms that appear in less than 2 documents
            max_df=0.95,         # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True    # Apply sublinear tf scaling
        )
        
        # Fit and transform the text data
        tfidf_features = self.vectorizer.fit_transform(df['text_clean'])
        
        # Create additional features
        additional_features = pd.DataFrame({
            'title_length': df['title'].str.len(),
            'abstract_length': df['abstract'].str.len(),
            'text_length': df['text_clean'].str.len(),
            'word_count': df['text_clean'].str.split().str.len(),
            'unique_words': df['text_clean'].apply(lambda x: len(set(x.split()))),
            'avg_word_length': df['text_clean'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
        })
        
        # Normalize additional features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        additional_features_scaled = scaler.fit_transform(additional_features)
        
        # Combine TF-IDF and additional features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, additional_features_scaled])
        
        # Prepare labels
        labels = self.label_encoder.fit_transform(df['category'])
        
        # Store feature names for interpretation
        tfidf_names = self.vectorizer.get_feature_names_out()
        additional_names = additional_features.columns.tolist()
        self.feature_names = list(tfidf_names) + additional_names
        
        print(f"Created {combined_features.shape[1]} features")
        print(f"TF-IDF features: {len(tfidf_names)}")
        print(f"Additional features: {len(additional_names)}")
        
        return combined_features, labels, self.feature_names
    
    def process_complete_pipeline(self, file_path):
        """
        Run the complete data processing pipeline
        
        Args:
            file_path (str): Path to the messy dataset
            
        Returns:
            tuple: (processed_features, labels, cleaned_dataframe)
        """
        print("Starting complete data processing pipeline...")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Step 1: Handle missing data
        df = self.handle_missing_data(df)
        
        # Step 2: Clean HTML and special characters
        print("Cleaning HTML and special characters...")
        df['title'] = df['title'].apply(self.clean_html_and_special_chars)
        df['abstract'] = df['abstract'].apply(self.clean_html_and_special_chars)
        
        # Step 3: Standardize categories
        df = self.standardize_categories(df)
        
        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 5: Advanced feature engineering
        features, labels, feature_names = self.advanced_feature_engineering(df)
        
        print("=" * 50)
        print("Data processing pipeline completed!")
        print(f"Final dataset size: {len(df)} rows")
        print(f"Number of features: {features.shape[1]}")
        print(f"Number of categories: {len(np.unique(labels))}")
        
        return features, labels, df

# Example usage
if __name__ == "__main__":
    preprocessor = AdvancedDataPreprocessor()
    
    # Process the messy dataset
    features, labels, clean_df = preprocessor.process_complete_pipeline('data/messy_research_papers.csv')
    
    print("\nPreprocessing completed successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
