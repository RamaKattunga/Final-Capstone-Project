import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class AdvancedNLPProcessor:
    def __init__(self):
        """Initialize the NLP processor with required tools"""
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Try to load spaCy model, fallback if not available
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model not found. Some features will be limited.")
            print("Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
    def advanced_text_preprocessing(self, text):
        """
        Perform comprehensive text preprocessing
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            dict: Preprocessed text in various formats
        """
        if not isinstance(text, str):
            text = str(text)
        
        # TODO: Implement comprehensive text preprocessing
        # Hints:
        # - Clean text (remove special characters, extra spaces)
        # - Tokenize into sentences and words
        # - Remove stopwords
        # - Perform lemmatization
        # - Extract different text representations
        
        # Basic cleaning
        text_clean = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        text_clean = re.sub(r'\s+', ' ', text_clean.strip())
        
        # Sentence tokenization
        sentences = nltk.sent_tokenize(text_clean)
        
        # Word tokenization and basic processing
        words = nltk.word_tokenize(text_clean.lower())
        words_filtered = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Advanced processing with spaCy (if available)
        if self.nlp:
            doc = self.nlp(text_clean)
            lemmatized_words = [token.lemma_.lower() for token in doc 
                              if token.is_alpha and not token.is_stop]
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [(token.text, token.pos_) for token in doc]
        else:
            # Fallback processing
            lemmatized_words = words_filtered  # Simple fallback
            named_entities = []
            pos_tags = nltk.pos_tag(words)
        
        return {
            'original': text,
            'cleaned': text_clean,
            'sentences': sentences,
            'words': words_filtered,
            'lemmatized': lemmatized_words,
            'entities': named_entities,
            'pos_tags': pos_tags,
            'sentence_count': len(sentences),
            'word_count': len(words_filtered),
            'avg_sentence_length': np.mean([len(nltk.word_tokenize(sent)) for sent in sentences]) if sentences else 0
        }
    
    def extract_text_statistics(self, text):
        """
        Extract comprehensive text statistics
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Text statistics
        """
        processed = self.advanced_text_preprocessing(text)
        
        # TODO: Calculate comprehensive text statistics
        # Hints:
        # - Basic counts (words, sentences, characters)
        # - Readability scores
        # - Vocabulary richness
        # - Average word/sentence lengths
        
        # Basic statistics
        char_count = len(processed['original'])
        word_count = processed['word_count']
        sentence_count = processed['sentence_count']
        
        # Advanced statistics
        unique_words = len(set(processed['words']))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Word length statistics
        word_lengths = [len(word) for word in processed['words']]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Readability scores
        try:
            flesch_score = flesch_reading_ease(processed['original'])
            flesch_grade = flesch_kincaid_grade(processed['original'])
        except:
            flesch_score = 0
            flesch_grade = 0
        
        # POS tag distribution
        pos_counts = Counter([tag for _, tag in processed['pos_tags']])
        
        return {
            'character_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'unique_words': unique_words,
            'vocabulary_richness': vocabulary_richness,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': processed['avg_sentence_length'],
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': flesch_grade,
            'pos_distribution': dict(pos_counts),
            'named_entities': processed['entities']
        }
    
    def extract_topics(self, documents, n_topics=5):
        """
        Extract topics from a collection of documents using LDA
        
        Args:
            documents (list): List of documents
            n_topics (int): Number of topics to extract
            
        Returns:
            dict: Topic information
        """
        print(f"Extracting {n_topics} topics from {len(documents)} documents...")
        
        # TODO: Implement topic extraction using LDA
        # Hints:
        # - Use CountVectorizer for LDA input
        # - Train LDA model
        # - Extract topic words and document-topic distributions
        
        # Prepare documents
        processed_docs = []
        for doc in documents:
            processed = self.advanced_text_preprocessing(doc)
            processed_docs.append(' '.join(processed['lemmatized']))
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            lowercase=True,
            min_df=2
        )
        
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Train LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic_weights
            })
        
        # Get document-topic distributions
        doc_topic_dist = lda.transform(doc_term_matrix)
        
        return {
            'topics': topics,
            'document_topic_distributions': doc_topic_dist,
            'vectorizer': vectorizer,
            'lda_model': lda
        }
    
    def analyze_document_collection(self, documents, labels=None):
        """
        Perform comprehensive analysis of a document collection
        
        Args:
            documents (list): List of documents to analyze
            labels (list): Optional category labels for documents
            
        Returns:
            dict: Comprehensive analysis results
        """
        print(f"Analyzing collection of {len(documents)} documents...")
        
        # Individual document analysis
        document_stats = []
        all_words = []
        all_entities = []
        
        for i, doc in enumerate(documents):
            stats = self.extract_text_statistics(doc)
            stats['document_id'] = i
            if labels:
                stats['category'] = labels[i]
            document_stats.append(stats)
            
            # Collect words and entities for collection-level analysis
            processed = self.advanced_text_preprocessing(doc)
            all_words.extend(processed['words'])
            all_entities.extend([ent[0] for ent in processed['entities']])
        
        # Collection-level statistics
        word_freq = Counter(all_words)
        entity_freq = Counter(all_entities)
        
        # Topic extraction
        topic_analysis = self.extract_topics(documents)
        
        # Category-specific analysis if labels provided
        category_analysis = {}
        if labels:
            categories = set(labels)
            for category in categories:
                category_docs = [doc for doc, label in zip(documents, labels) if label == category]
                category_stats = []
                
                for doc in category_docs:
                    stats = self.extract_text_statistics(doc)
                    category_stats.append(stats)
                
                # Calculate category averages
                avg_stats = {}
                numeric_fields = ['word_count', 'sentence_count', 'vocabulary_richness', 
                                'avg_word_length', 'flesch_reading_ease']
                
                for field in numeric_fields:
                    values = [stats[field] for stats in category_stats if field in stats]
                    avg_stats[f'avg_{field}'] = np.mean(values) if values else 0
                
                category_analysis[category] = {
                    'document_count': len(category_docs),
                    'average_statistics': avg_stats,
                    'individual_statistics': category_stats
                }
        
        return {
            'document_statistics': document_stats,
            'collection_statistics': {
                'total_documents': len(documents),
                'total_unique_words': len(word_freq),
                'most_common_words': word_freq.most_common(20),
                'most_common_entities': entity_freq.most_common(10),
                'average_document_length': np.mean([stats['word_count'] for stats in document_stats]),
                'average_readability': np.mean([stats['flesch_reading_ease'] for stats in document_stats])
            },
            'topic_analysis': topic_analysis,
            'category_analysis': category_analysis
        }

# Example usage
if __name__ == "__main__":
    processor = AdvancedNLPProcessor()
    
    # Test with sample text
    sample_text = """
    This is a sample research paper about machine learning applications in healthcare.
    The study demonstrates significant improvements in diagnostic accuracy through the use of
    advanced neural network architectures. The research has important implications for
    clinical practice and patient outcomes.
    """
    
    # Analyze sample text
    stats = processor.extract_text_statistics(sample_text)
    print("Text Statistics:")
    for key, value in stats.items():
        if key != 'pos_distribution':
            print(f"{key}: {value}")
    
    print("\nNLP Processor initialized successfully!")
