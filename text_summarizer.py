import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict, Counter
import re

class TextSummarizer:
    def __init__(self):
        """Initialize the text summarization system"""
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
    def preprocess_text_for_summarization(self, text):
        """
        Preprocess text specifically for summarization
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            tuple: (sentences, cleaned_sentences)
        """
        # TODO: Implement text preprocessing for summarization
        # Hints:
        # - Split into sentences
        # - Clean sentences (remove extra spaces, special characters)
        # - Remove very short sentences
        # - Keep original sentences for final summary
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        original_sentences = []
        
        for sentence in sentences:
            # Clean the sentence
            cleaned = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned.strip())
            
            # Keep sentences with reasonable length
            if len(cleaned.split()) >= 5:  # At least 5 words
                cleaned_sentences.append(cleaned.lower())
                original_sentences.append(sentence.strip())
        
        return original_sentences, cleaned_sentences
    
    def calculate_sentence_scores(self, sentences, method='tfidf'):
        """
        Calculate importance scores for sentences
        
        Args:
            sentences (list): List of sentences
            method (str): Scoring method ('tfidf', 'frequency', 'position')
            
        Returns:
            list: Sentence scores
        """
        # TODO: Implement different sentence scoring methods
        # Hints:
        # - TF-IDF based scoring
        # - Word frequency based scoring  
        # - Position-based scoring (first/last sentences often important)
        # - Combine multiple scoring methods
        
        if method == 'tfidf':
            return self._tfidf_scoring(sentences)
        elif method == 'frequency':
            return self._frequency_scoring(sentences)
        elif method == 'position':
            return self._position_scoring(sentences)
        elif method == 'combined':
            return self._combined_scoring(sentences)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def _tfidf_scoring(self, sentences):
        """TF-IDF based sentence scoring"""
        if len(sentences) < 2:
            return [1.0] * len(sentences)
        
        # Create TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores as sum of TF-IDF values
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            if scores.max() > 0:
                scores = scores / scores.max()
            
        except ValueError:
            # Fallback for edge cases
            scores = np.ones(len(sentences))
        
        return scores.tolist()
    
    def _frequency_scoring(self, sentences):
        """Frequency-based sentence scoring"""
        # Count word frequencies
        word_freq = defaultdict(int)
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    word_freq[word] += 1
        
        # Calculate sentence scores
        scores = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            sentence_score = 0
            word_count = 0
            
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    sentence_score += word_freq[word]
                    word_count += 1
            
            # Average frequency score
            if word_count > 0:
                scores.append(sentence_score / word_count)
            else:
                scores.append(0)
        
        # Normalize scores
        if max(scores) > 0:
            scores = [score / max(scores) for score in scores]
        
        return scores
    
    def _position_scoring(self, sentences):
        """Position-based sentence scoring"""
        scores = []
        n = len(sentences)
        
        for i in range(n):
            # First and last sentences get higher scores
            if i == 0 or i == n-1:
                score = 1.0
            elif i < n * 0.1:  # First 10%
                score = 0.8
            elif i > n * 0.9:  # Last 10%
                score = 0.8
            else:
                # Middle sentences get lower scores, but not zero
                score = 0.3
            
            scores.append(score)
        
        return scores
    
    def _combined_scoring(self, sentences):
        """Combined scoring using multiple methods"""
        tfidf_scores = self._tfidf_scoring(sentences)
        freq_scores = self._frequency_scoring(sentences)
        pos_scores = self._position_scoring(sentences)
        
        # Weighted combination
        combined_scores = []
        for i in range(len(sentences)):
            combined = (0.5 * tfidf_scores[i] + 
                       0.3 * freq_scores[i] + 
                       0.2 * pos_scores[i])
            combined_scores.append(combined)
        
        return combined_scores
    
    def extractive_summarization(self, text, summary_ratio=0.3, max_sentences=5, method='combined'):
        """
        Generate extractive summary by selecting important sentences
        
        Args:
            text (str): Input text to summarize
            summary_ratio (float): Ratio of sentences to keep (0.1 to 0.5)
            max_sentences (int): Maximum number of sentences in summary
            method (str): Scoring method to use
            
        Returns:
            dict: Summary information
        """
        print(f"Generating extractive summary (ratio: {summary_ratio}, method: {method})...")
        
        # Preprocess text
        original_sentences, cleaned_sentences = self.preprocess_text_for_summarization(text)
        
        if len(original_sentences) == 0:
            return {
                'summary': "No content available for summarization.",
                'summary_sentences': [],
                'original_sentence_count': 0,
                'summary_sentence_count': 0,
                'compression_ratio': 0,
                'selected_indices': []
            }
        
        # Calculate sentence scores
        scores = self.calculate_sentence_scores(cleaned_sentences, method=method)
        
        # Determine number of sentences for summary
        num_sentences = min(
            max_sentences,
            max(1, int(len(original_sentences) * summary_ratio))
        )
        
        # TODO: Select top sentences for summary
        # Hints:
        # - Get indices of highest scoring sentences
        # - Sort selected sentences by original order
        # - Combine sentences into final summary
        
        # Get indices of top sentences
        sentence_scores = list(enumerate(scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, score in sentence_scores[:num_sentences]]
        
        # Sort by original order to maintain coherence
        selected_indices.sort()
        
        # Create summary
        summary_sentences = [original_sentences[idx] for idx in selected_indices]
        summary_text = ' '.join(summary_sentences)
        
        # Calculate metrics
        compression_ratio = len(summary_sentences) / len(original_sentences)
        
        return {
            'summary': summary_text,
            'summary_sentences': summary_sentences,
            'original_sentence_count': len(original_sentences),
            'summary_sentence_count': len(summary_sentences),
            'compression_ratio': compression_ratio,
            'selected_indices': selected_indices,
            'sentence_scores': scores,
            'method_used': method
        }
    
    def abstractive_summarization_simple(self, text, max_length=100):
        """
        Simple abstractive summarization using keyword extraction and template generation
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary in words
            
        Returns:
            dict: Abstractive summary information
        """
        print("Generating simple abstractive summary...")
        
        # TODO: Implement basic abstractive summarization
        # Hints:
        # - Extract key terms and phrases
        # - Identify main topics
        # - Generate new sentences using templates
        # - This is a simplified version - real abstractive summarization uses deep learning
        
        # Preprocess text
        original_sentences, cleaned_sentences = self.preprocess_text_for_summarization(text)
        
        # Extract key terms
        all_text = ' '.join(cleaned_sentences)
        words = nltk.word_tokenize(all_text.lower())
        words_filtered = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Get most frequent terms
        word_freq = Counter(words_filtered)
        key_terms = [word for word, freq in word_freq.most_common(10)]
        
        # Extract key phrases (simple bigrams)
        bigrams = list(nltk.bigrams(words_filtered))
        bigram_freq = Counter(bigrams)
        key_phrases = [' '.join(bigram) for bigram, freq in bigram_freq.most_common(5)]
        
        # Simple template-based summary generation
        if len(key_terms) >= 3:
            summary = f"This document discusses {key_terms[0]}, {key_terms[1]}, and {key_terms[2]}."
            if key_phrases:
                summary += f" Key topics include {key_phrases[0]}."
            if len(key_terms) > 5:
                summary += f" The research also covers {key_terms[4]} and related concepts."
        else:
            summary = "This document contains research findings and analysis."
        
        # Trim to max length
        summary_words = summary.split()
        if len(summary_words) > max_length:
            summary = ' '.join(summary_words[:max_length]) + "..."
        
        return {
            'summary': summary,
            'key_terms': key_terms,
            'key_phrases': key_phrases,
            'word_count': len(summary.split()),
            'method': 'template_based'
        }
    
    def multi_document_summarization(self, documents, labels=None, summary_ratio=0.2):
        """
        Generate summaries for multiple documents
        
        Args:
            documents (list): List of documents to summarize
            labels (list): Optional labels for documents
            summary_ratio (float): Ratio for individual summaries
            
        Returns:
            dict: Multi-document summary results
        """
        print(f"Generating summaries for {len(documents)} documents...")
        
        summaries = []
        all_summaries_text = []
        
        for i, doc in enumerate(documents):
            # Generate individual summary
            summary_result = self.extractive_summarization(
                doc, 
                summary_ratio=summary_ratio, 
                method='combined'
            )
            
            # Add document info
            summary_result['document_id'] = i
            if labels and i < len(labels):
                summary_result['category'] = labels[i]
            
            summaries.append(summary_result)
            all_summaries_text.append(summary_result['summary'])
        
        # Generate overall summary from individual summaries
        combined_text = ' '.join(all_summaries_text)
        overall_summary = self.extractive_summarization(
            combined_text, 
            summary_ratio=0.3, 
            max_sentences=10,
            method='combined'
        )
        
        return {
            'individual_summaries': summaries,
            'overall_summary': overall_summary,
            'total_documents': len(documents),
            'average_compression_ratio': np.mean([s['compression_ratio'] for s in summaries])
        }
    
    def evaluate_summary_quality(self, original_text, summary_text):
        """
        Evaluate summary quality using various metrics
        
        Args:
            original_text (str): Original text
            summary_text (str): Generated summary
            
        Returns:
            dict: Quality metrics
        """
        # TODO: Implement summary quality evaluation
        # Hints:
        # - Coverage: How much of original content is covered
        # - Compression ratio
        # - Readability scores
        # - Coherence measures
        
        # Basic metrics
        original_words = len(original_text.split())
        summary_words = len(summary_text.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        # Coverage analysis (simple word overlap)
        original_words_set = set(original_text.lower().split())
        summary_words_set = set(summary_text.lower().split())
        coverage = len(original_words_set.intersection(summary_words_set)) / len(original_words_set) if original_words_set else 0
        
        # Readability
        try:
            from textstat import flesch_reading_ease
            readability = flesch_reading_ease(summary_text)
        except:
            readability = 0
        
        return {
            'compression_ratio': compression_ratio,
            'coverage_ratio': coverage,
            'readability_score': readability,
            'original_word_count': original_words,
            'summary_word_count': summary_words,
            'quality_score': (coverage + (1 - compression_ratio)) / 2  # Simple quality metric
        }

# Example usage
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    # Test with sample text
    sample_text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
    build mathematical models based on training data in order to make predictions or decisions without 
    being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of 
    applications, such as email filtering and computer vision, where it is difficult or infeasible 
    to develop conventional algorithms to perform the needed tasks.
    """
    
    # Generate summary
    result = summarizer.extractive_summarization(sample_text, summary_ratio=0.5)
    print("Summary:")
    print(result['summary'])
    print(f"Compression: {result['compression_ratio']:.2f}")
    
    print("\nText Summarizer initialized successfully!")
