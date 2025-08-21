"""
Document Classification System - Main Execution Script

This script runs the complete document classification pipeline:
1. Load and process data
2. Train classification models
3. Evaluate model performance
4. Save results and models

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
from src.data_processor import DataProcessor
from src.classifier import DocumentClassifier
from src.evaluator import ModelEvaluator

def main():
    """Main function to run the document classification system"""
    
    print("=" * 60)
    print("DOCUMENT CLASSIFICATION SYSTEM")
    print("=" * 60)
    
    # Create results folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Step 1: Load and Process Data
    print("\n1. Loading and Processing Data...")
    processor = DataProcessor()
    
    # Load dataset
    data = processor.load_data('data/research_papers_dataset.csv')
    
    if data is None:
        print("Error: Could not load dataset. Please check the file path.")
        return
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = processor.prepare_data(data)
    
    # Get unique categories
    categories = sorted(data['category'].unique())
    print(f"Categories found: {categories}")
    
    # Step 2: Train Classification Models
    print("\n2. Training Classification Models...")
    classifier = DocumentClassifier()
    classifier.train_models(X_train, y_train)
    
    # Save trained models
    classifier.save_models()
    
    # Step 3: Evaluate Models
    print("\n3. Evaluating Model Performance...")
    evaluator = ModelEvaluator()
    
    # Test each model
    model_names = ['linear', 'neural', 'ensemble']
    
    for model_name in model_names:
        # Make predictions
        y_pred = classifier.predict(X_test, model_name)
        
        # Evaluate performance
        evaluator.evaluate_model(y_test, y_pred, model_name)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(y_test, y_pred, model_name, categories)
    
    # Step 4: Compare Models
    print("\n4. Comparing All Models...")
    comparison_df = evaluator.compare_models()
    
    # Plot comparison
    evaluator.plot_model_comparison(comparison_df)
    
    # Save comparison results
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    # Step 5: Test with Sample Predictions
    print("\n5. Testing with Sample Predictions...")
    test_sample_predictions(classifier, processor, categories)
    
    print("\n" + "=" * 60)
    print("DOCUMENT CLASSIFICATION COMPLETED!")
    print("Check the 'results/' folder for output files.")
    print("Check the 'models/' folder for saved models.")
    print("=" * 60)

def test_sample_predictions(classifier, processor, categories):
    """
    Test the classifier with sample text
    
    Args:
        classifier: Trained classifier
        processor: Data processor
        categories: List of categories
    """
    # Sample texts for testing
    sample_texts = [
        "Machine learning algorithms for deep neural networks and artificial intelligence",
        "Clinical trials for cancer treatment and pharmaceutical research",
        "Stock market analysis and financial investment strategies",
        "Online learning platforms and educational technology innovations",
        "Climate change and environmental sustainability research"
    ]
    
    expected_categories = [
        "Technology", "Healthcare", "Finance", "Education", "Environment"
    ]
    
    print("\nTesting with sample texts:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts):
        # Clean and process the text
        cleaned_text = processor.clean_text(text)
        
        # Convert to the same format as training data
        text_vectorized = processor.vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = classifier.predict(text_vectorized, 'ensemble')[0]
        probabilities = classifier.predict_proba(text_vectorized, 'ensemble')[0]
        
        # Find the confidence (highest probability)
        max_prob = max(probabilities)
        
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"Predicted: {prediction} (Confidence: {max_prob:.2f})")
        print(f"Expected: {expected_categories[i]}")
        
        # Show all probabilities
        prob_dict = dict(zip(categories, probabilities))
        print("All probabilities:")
        for cat, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {prob:.3f}")

if __name__ == "__main__":
    main()
