"""
Simple demo script to test NLP functionality
"""

from simple_nlp import SimpleNLPAnalyzer
import pandas as pd

def demo_single_paper():
    """Demo analysis of a single paper"""
    print("=" * 50)
    print("DEMO: Single Paper Analysis")
    print("=" * 50)
    
    # Create analyzer
    nlp = SimpleNLPAnalyzer()
    
    # Sample papers for testing
    test_papers = [
        {
            'title': 'Deep Learning for Medical Image Analysis',
            'abstract': 'This research explores the application of deep learning techniques for analyzing medical images. Our study shows significant improvements in diagnostic accuracy using convolutional neural networks for detecting abnormalities in X-ray images.',
            'category': 'Healthcare'
        },
        {
            'title': 'Climate Change Impact on Ocean Temperatures',
            'abstract': 'We present a comprehensive analysis of rising ocean temperatures due to climate change. Our findings indicate alarming trends that could severely impact marine ecosystems and global weather patterns.',
            'category': 'Environment'
        },
        {
            'title': 'Machine Learning in Financial Trading',
            'abstract': 'This paper demonstrates how machine learning algorithms can be successfully applied to financial trading strategies. We achieve consistent profits using ensemble methods for market prediction.',
            'category': 'Finance'
        }
    ]
    
    # Analyze each test paper
    for i, paper in enumerate(test_papers):
        print(f"\nPaper {i+1}: {paper['title']}")
        print("-" * 40)
        
        result = nlp.analyze_single_paper(
            paper['title'], 
            paper['abstract'], 
            paper['category']
        )
        
        print(f"Category: {result['category']}")
        print(f"Word Count: {result['word_count']}")
        print(f"Key Phrases: {', '.join(result['key_phrases'])}")
        print(f"Summary: {result['summary']}")
        print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")

def demo_dataset_analysis():
    """Demo analysis of full dataset"""
    print("\n" + "=" * 50)
    print("DEMO: Full Dataset Analysis")
    print("=" * 50)
    
    # Check if cleaned dataset exists
    dataset_path = 'results/component2_results/cleaned_dataset.csv'
    
    try:
        # Load a small sample for demo
        df = pd.read_csv(dataset_path)
        print(f"Found dataset with {len(df)} papers")
        
        # Take small sample for demo (first 20 papers)
        sample_df = df.head(20)
        print(f"Analyzing sample of {len(sample_df)} papers for demo...")
        
        # Create analyzer and analyze sample
        nlp = SimpleNLPAnalyzer()
        results = nlp.analyze_dataset_sample(sample_df)
        
        # Show sample results
        print("\nSample Results:")
        print(results[['title', 'category', 'sentiment_label', 'word_count']].head())
        
    except FileNotFoundError:
        print("Dataset not found. Please run Components 1 and 2 first.")
        print("Or use the single paper demo above.")

if __name__ == "__main__":
    # Run demos
    demo_single_paper()
    demo_dataset_analysis()
