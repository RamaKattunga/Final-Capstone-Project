import pandas as pd
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import matplotlib.pyplot as plt
from collections import Counter
import re

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class SimpleNLPAnalyzer:
    def __init__(self):
        """Simple NLP analyzer for research papers"""
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Simple text cleaning
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # TODO: Clean the text
        # Hint: Remove special characters, convert to lowercase
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep letters, numbers, spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def extract_key_phrases(self, text, num_phrases=5):
        """
        Extract key phrases from text
        
        Args:
            text (str): Text to analyze
            num_phrases (int): Number of key phrases to extract
            
        Returns:
            list: List of key phrases
        """
        if not text:
            return []
        
        # TODO: Extract important phrases
        # Hint: Use word frequency, remove stop words
        
        # Clean text and split into words
        words = self.clean_text(text).split()
        
        # Remove stop words and short words
        important_words = [word for word in words 
                          if word not in self.stop_words and len(word) > 3]
        
        # Count word frequency
        word_freq = Counter(important_words)
        
        # Get most common words as key phrases
        key_phrases = [word for word, count in word_freq.most_common(num_phrases)]
        
        return key_phrases
    
    def create_summary(self, text, num_sentences=2):
        """
        Create extractive summary of text
        
        Args:
            text (str): Text to summarize
            num_sentences (int): Number of sentences in summary
            
        Returns:
            str: Summary text
        """
        if not text or len(text.split()) < 10:
            return text
        
        try:
            # TODO: Create automatic summary
            # Hint: Use the Sumy library for extractive summarization
            
            # Use Sumy for extractive summarization
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            
            # Generate summary
            summary_sentences = summarizer(parser.document, num_sentences)
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            return summary
            
        except:
            # Fallback: return first few sentences
            sentences = text.split('.')[:num_sentences]
            return '. '.join(sentences) + '.'
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text:
            return {'polarity': 0, 'sentiment': 'neutral', 'confidence': 0}
        
        # TODO: Analyze sentiment using TextBlob
        # Hint: TextBlob gives polarity from -1 (negative) to +1 (positive)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence (how far from neutral)
        confidence = abs(polarity)
        
        return {
            'polarity': polarity,
            'sentiment': sentiment,
            'confidence': confidence
        }
    
    def analyze_single_paper(self, title, abstract, category):
        """
        Analyze a single research paper
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            category (str): Paper category
            
        Returns:
            dict: Complete analysis results
        """
        # Combine title and abstract
        full_text = f"{title} {abstract}"
        
        # TODO: Run all analyses on the paper
        # Hint: Use the methods defined above
        
        # Run all analyses
        key_phrases = self.extract_key_phrases(full_text, 5)
        summary = self.create_summary(abstract, 2)
        sentiment = self.analyze_sentiment(full_text)
        
        # Calculate text statistics
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        return {
            'title': title,
            'category': category,
            'word_count': word_count,
            'char_count': char_count,
            'key_phrases': key_phrases,
            'summary': summary,
            'sentiment_score': sentiment['polarity'],
            'sentiment_label': sentiment['sentiment'],
            'sentiment_confidence': sentiment['confidence']
        }
    
    def analyze_dataset(self, csv_file_path):
        """
        Analyze entire dataset of research papers
        
        Args:
            csv_file_path (str): Path to CSV file with papers
            
        Returns:
            pandas.DataFrame: Analysis results for all papers
        """
        print(f"Loading dataset from: {csv_file_path}")
        
        # Load the dataset
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} papers")
        
        # Analyze each paper
        results = []
        
        for i, row in df.iterrows():
            if i % 50 == 0:  # Progress update every 50 papers
                print(f"Analyzing paper {i+1}/{len(df)}")
            
            # TODO: Analyze each paper and store results
            # Hint: Use analyze_single_paper method
            
            try:
                analysis = self.analyze_single_paper(
                    row['title'], 
                    row['abstract'], 
                    row['category']
                )
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing paper {i}: {e}")
                # Add empty result to maintain order
                results.append({
                    'title': row['title'],
                    'category': row['category'],
                    'word_count': 0,
                    'char_count': 0,
                    'key_phrases': [],
                    'summary': '',
                    'sentiment_score': 0,
                    'sentiment_label': 'neutral',
                    'sentiment_confidence': 0
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        print("Analysis completed!")
        return results_df
    
    def create_visualizations(self, results_df, save_folder='results/nlp_results/'):
        """
        Create simple visualizations of the analysis
        
        Args:
            results_df (pandas.DataFrame): Analysis results
            save_folder (str): Folder to save visualizations
        """
        import os
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # TODO: Create visualizations
        # Hint: Use matplotlib for simple charts
        
        # 1. Sentiment by Category
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Average sentiment by category
        plt.subplot(2, 2, 1)
        category_sentiment = results_df.groupby('category')['sentiment_score'].mean()
        plt.bar(category_sentiment.index, category_sentiment.values)
        plt.title('Average Sentiment by Category')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        
        # Plot 2: Sentiment distribution
        plt.subplot(2, 2, 2)
        sentiment_counts = results_df['sentiment_label'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Overall Sentiment Distribution')
        
        # Plot 3: Word count by category
        plt.subplot(2, 2, 3)
        category_words = results_df.groupby('category')['word_count'].mean()
        plt.bar(category_words.index, category_words.values)
        plt.title('Average Word Count by Category')
        plt.ylabel('Word Count')
        plt.xticks(rotation=45)
        
        # Plot 4: Sentiment confidence
        plt.subplot(2, 2, 4)
        plt.hist(results_df['sentiment_confidence'], bins=20, alpha=0.7)
        plt.title('Sentiment Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Papers')
        
        plt.tight_layout()
        plt.savefig(f'{save_folder}/nlp_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Word Cloud of Key Phrases
        print("Creating word cloud...")
        all_phrases = []
        for phrases in results_df['key_phrases']:
            if isinstance(phrases, list):
                all_phrases.extend(phrases)
        
        if all_phrases:
            phrase_text = ' '.join(all_phrases)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(phrase_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Key Phrases Across All Papers')
            plt.savefig(f'{save_folder}/key_phrases_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self, results_df, save_folder='results/nlp_results/'):
        """
        Generate a simple summary report
        
        Args:
            results_df (pandas.DataFrame): Analysis results
            save_folder (str): Folder to save report
        """
        # TODO: Create summary statistics
        # Hint: Calculate averages, counts, and interesting findings
        
        total_papers = len(results_df)
        avg_sentiment = results_df['sentiment_score'].mean()
        most_positive_category = results_df.groupby('category')['sentiment_score'].mean().idxmax()
        most_negative_category = results_df.groupby('category')['sentiment_score'].mean().idxmin()
        
        # Get most common key phrases
        all_phrases = []
        for phrases in results_df['key_phrases']:
            if isinstance(phrases, list):
                all_phrases.extend(phrases)
        
        common_phrases = Counter(all_phrases).most_common(10)
        
        # Create report
        report = f"""
# NLP Analysis Report

## Dataset Summary
- Total papers analyzed: {total_papers}
- Average sentiment score: {avg_sentiment:.3f}
- Most positive category: {most_positive_category}
- Most negative category: {most_negative_category}

## Sentiment Distribution
{results_df['sentiment_label'].value_counts().to_string()}

## Category Analysis
{results_df.groupby('category').agg({
    'sentiment_score': 'mean',
    'word_count': 'mean',
    'sentiment_confidence': 'mean'
}).round(3).to_string()}

## Top 10 Key Phrases Across All Papers
{chr(10).join([f"{i+1}. {phrase} ({count} times)" for i, (phrase, count) in enumerate(common_phrases)])}

## Sample Summaries by Category
"""
        
        # Add sample summaries for each category
        for category in results_df['category'].unique():
            category_data = results_df[results_df['category'] == category]
            if not category_data.empty:
                sample = category_data.iloc[0]
                report += f"\n### {category} Example:\n"
                report += f"**Title:** {sample['title']}\n"
                report += f"**Summary:** {sample['summary']}\n"
                report += f"**Key Phrases:** {', '.join(sample['key_phrases'][:3])}\n"
                report += f"**Sentiment:** {sample['sentiment_label']} ({sample['sentiment_score']:.3f})\n"
        
        # Save report
        import os
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        with open(f'{save_folder}/nlp_analysis_report.md', 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {save_folder}/nlp_analysis_report.md")
        return report

# Example usage
if __name__ == "__main__":
    # Create analyzer
    nlp = SimpleNLPAnalyzer()
    
    # Test with a sample paper
    sample_title = "Machine Learning for Healthcare Applications"
    sample_abstract = "This paper presents a comprehensive study of machine learning applications in healthcare. We explore various algorithms and their effectiveness in medical diagnosis and treatment prediction."
    
    result = nlp.analyze_single_paper(sample_title, sample_abstract, "Healthcare")
    print("Sample Analysis Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
