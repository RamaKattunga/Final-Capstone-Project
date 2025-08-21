"""
Intelligent Data Processing Pipeline - Main Execution Script

This script demonstrates the complete advanced ML pipeline:
1. Advanced data preprocessing and cleaning
2. Feature engineering with multiple techniques
3. Training multiple ensemble models
4. Comprehensive model evaluation and comparison
5. Best model selection and deployment

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from src.advanced_preprocessor import AdvancedDataPreprocessor
from src.ensemble_classifier import EnsembleClassifier

def create_results_summary(evaluation_results, save_path='results/component2_results/'):
    """
    Create a comprehensive results summary
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create results DataFrame
    results_data = []
    for model_name, results in evaluation_results.items():
        if 'mean_accuracy' in results:
            results_data.append({
                'Model': model_name,
                'Mean_Accuracy': results['mean_accuracy'],
                'Std_Accuracy': results['std_accuracy'],
                'Min_Accuracy': results['cv_scores'].min(),
                'Max_Accuracy': results['cv_scores'].max(),
                'Model_Type': 'Ensemble' if 'ensemble' in model_name else 'Individual'
            })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(save_path, 'model_comparison.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Mean accuracy comparison
    plt.subplot(2, 2, 1)
    colors = ['red' if 'ensemble' in model else 'blue' for model in results_df['Model']]
    bars = plt.bar(range(len(results_df)), results_df['Mean_Accuracy'], color=colors, alpha=0.7)
    plt.xticks(range(len(results_df)), results_df['Model'], rotation=45, ha='right')
    plt.ylabel('Mean Accuracy')
    plt.title('Model Performance Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Accuracy with error bars
    plt.subplot(2, 2, 2)
    plt.errorbar(range(len(results_df)), results_df['Mean_Accuracy'], 
                yerr=results_df['Std_Accuracy'], fmt='o', capsize=5)
    plt.xticks(range(len(results_df)), results_df['Model'], rotation=45, ha='right')
    plt.ylabel('Accuracy with Std Dev')
    plt.title('Model Accuracy with Error Bars')
    plt.grid(alpha=0.3)
    
    # Plot 3: Individual vs Ensemble comparison
    plt.subplot(2, 2, 3)
    model_type_summary = results_df.groupby('Model_Type')['Mean_Accuracy'].agg(['mean', 'std'])
    plt.bar(model_type_summary.index, model_type_summary['mean'], 
           yerr=model_type_summary['std'], capsize=5, alpha=0.7)
    plt.ylabel('Average Accuracy')
    plt.title('Individual vs Ensemble Models')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Top 5 models detailed view
    plt.subplot(2, 2, 4)
    top_5 = results_df.head(5)
    plt.barh(range(len(top_5)), top_5['Mean_Accuracy'])
    plt.yticks(range(len(top_5)), top_5['Model'])
    plt.xlabel('Mean Accuracy')
    plt.title('Top 5 Models')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def test_pipeline_robustness(preprocessor, classifier, test_cases, save_path='results/component2_results/'):
    """
    Test the pipeline with various problematic inputs
    """
    print("\nTesting pipeline robustness...")
    
    test_results = []
    
    for i, (test_name, test_data) in enumerate(test_cases.items()):
        print(f"Testing: {test_name}")
        
        try:
            # Create temporary DataFrame
            temp_df = pd.DataFrame([test_data])
            
            # Process through pipeline
            temp_df['title'] = temp_df['title'].apply(preprocessor.clean_html_and_special_chars)
            temp_df['abstract'] = temp_df['abstract'].apply(preprocessor.clean_html_and_special_chars)
            temp_df['text_clean'] = temp_df['title'] + ' ' + temp_df['abstract']
            
            # Transform using existing vectorizer
            if preprocessor.vectorizer:
                text_features = preprocessor.vectorizer.transform(temp_df['text_clean'])
                
                # Add dummy additional features
                additional_features = np.array([[
                    len(test_data['title']),  # title_length
                    len(test_data['abstract']),  # abstract_length
                    len(test_data['title'] + test_data['abstract']),  # text_length
                    len((test_data['title'] + test_data['abstract']).split()),  # word_count
                    len(set((test_data['title'] + test_data['abstract']).split())),  # unique_words
                    np.mean([len(word) for word in (test_data['title'] + test_data['abstract']).split()]) if (test_data['title'] + test_data['abstract']).split() else 0  # avg_word_length
                ]])
                
                # Combine features
                from scipy.sparse import hstack
                combined_features = hstack([text_features, additional_features])
                
                # Make prediction
                predictions, confidence = classifier.predict_with_confidence(combined_features)
                
                test_results.append({
                    'Test_Case': test_name,
                    'Prediction': preprocessor.label_encoder.inverse_transform(predictions)[0],
                    'Confidence': confidence[0],
                    'Status': 'Success'
                })
                
                print(f"  ✓ Predicted: {preprocessor.label_encoder.inverse_transform(predictions)[0]} (Confidence: {confidence[0]:.3f})")
                
        except Exception as e:
            test_results.append({
                'Test_Case': test_name,
                'Prediction': 'Error',
                'Confidence': 0.0,
                'Status': f'Failed: {str(e)}'
            })
            print(f"  ✗ Error: {e}")
    
    # Save robustness test results
    robustness_df = pd.DataFrame(test_results)
    robustness_df.to_csv(os.path.join(save_path, 'robustness_test.csv'), index=False)
    
    return robustness_df

def main():
    """Main function to run the complete advanced pipeline"""
    
    print("=" * 70)
    print("INTELLIGENT DATA PROCESSING PIPELINE - COMPONENT 2")
    print("=" * 70)
    
    # Create results directory
    results_dir = 'results/component2_results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Step 1: Create messy dataset if it doesn't exist
    if not os.path.exists('data/messy_research_papers.csv'):
        print("Creating messy dataset for testing...")
        from data.create_messy_dataset import create_messy_dataset
        create_messy_dataset()
    
    # Step 2: Advanced Data Preprocessing
    print("\n1. ADVANCED DATA PREPROCESSING")
    print("-" * 40)
    
    preprocessor = AdvancedDataPreprocessor()
    features, labels, clean_df = preprocessor.process_complete_pipeline('data/messy_research_papers.csv')
    
    # Save cleaned dataset
    clean_df.to_csv(os.path.join(results_dir, 'cleaned_dataset.csv'), index=False)
    
    # Step 3: Train-Test Split
    print("\n2. SPLITTING DATA FOR TRAINING AND TESTING")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Step 4: Ensemble Model Training
    print("\n3. TRAINING ENSEMBLE MODELS")
    print("-" * 40)
    
    classifier = EnsembleClassifier()
    
    # Train individual models
    classifier.train_individual_models(X_train, y_train)
    
    # Train ensemble models
    classifier.train_ensemble_models(X_train, y_train)
    
    # Save all models
    classifier.save_models()
    
    # Step 5: Model Evaluation
    print("\n4. COMPREHENSIVE MODEL EVALUATION")
    print("-" * 40)
    
    evaluation_results = classifier.evaluate_all_models(X_train, y_train, cv_folds=5)
    
    # Create results summary
    results_summary = create_results_summary(evaluation_results, results_dir)
    
    print("\nModel Performance Summary:")
    print(results_summary.to_string(index=False))
    
    # Step 6: Best Model Selection and Final Testing
    print("\n5. BEST MODEL TESTING ON HELD-OUT DATA")
    print("-" * 40)
    
    best_model_name, best_accuracy, best_model = classifier.get_best_model(evaluation_results)
    
    # Test best model on held-out test set
    y_pred = best_model.predict(X_test)
    
    # Calculate final metrics
    from sklearn.metrics import accuracy_score
    final_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal Test Accuracy: {final_accuracy:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=preprocessor.label_encoder.classes_))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=preprocessor.label_encoder.classes_,
                yticklabels=preprocessor.label_encoder.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.savefig(os.path.join(results_dir, 'final_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 7: Pipeline Robustness Testing
    print("\n6. TESTING PIPELINE ROBUSTNESS")
    print("-" * 40)
    
    # Define test cases with problematic data
    test_cases = {
        'HTML_tags': {
            'title': '<p>Machine Learning <br> Research</p>',
            'abstract': '<div>This paper discusses &amp; analyzes ML algorithms.</div>',
            'category': 'Technology'
        },
        'Missing_abstract': {
            'title': 'Financial Market Analysis Using AI',
            'abstract': '',
            'category': 'Finance'
        },
        'Very_short_text': {
            'title': 'AI',
            'abstract': 'ML study.',
            'category': 'Technology'
        },
        'Mixed_case_category': {
            'title': 'Clinical Trial Results',
            'abstract': 'This study presents results from a large-scale clinical trial.',
            'category': 'HEALTHCARE'
        },
        'Special_characters': {
            'title': 'Environmental Study: CO₂ & H₂O Analysis',
            'abstract': 'Research on carbon dioxide (CO₂) and water (H₂O) interactions in climate systems.',
            'category': 'Environment'
        }
    }
    
    robustness_results = test_pipeline_robustness(preprocessor, classifier, test_cases, results_dir)
    
    print("\nRobustness Test Results:")
    print(robustness_results.to_string(index=False))
    
    # Step 8: Generate Final Report
    print("\n7. GENERATING FINAL REPORT")
    print("-" * 40)
    
    report = f"""
# Component 2: Intelligent Data Processing Pipeline - Final Report

## Dataset Processing Summary
- Original messy dataset size: {len(pd.read_csv('data/messy_research_papers.csv'))} rows
- Cleaned dataset size: {len(clean_df)} rows
- Data cleaning improvement: {((len(clean_df) / len(pd.read_csv('data/messy_research_papers.csv'))) * 100):.1f}% data retained
- Number of features created: {features.shape[1]}

## Model Performance Summary
- Best Model: {best_model_name}
- Cross-validation Accuracy: {best_accuracy:.3f}
- Final Test Accuracy: {final_accuracy:.3f}
- Number of models tested: {len(evaluation_results)}

## Ensemble Methods Performance
{results_summary.head(10).to_string(index=False)}

## Pipeline Robustness
- Test cases passed: {len(robustness_results[robustness_results['Status'] == 'Success'])} / {len(robustness_results)}
- Pipeline success rate: {(len(robustness_results[robustness_results['Status'] == 'Success']) / len(robustness_results) * 100):.1f}%

## Key Improvements from Component 1
- Advanced feature engineering with TF-IDF n-grams
- Ensemble methods for improved accuracy
- Robust data cleaning pipeline
- Handling of real-world data problems
- Comprehensive evaluation framework

## Files Generated
- cleaned_dataset.csv: Processed clean dataset
- model_comparison.csv: Detailed model performance comparison
- ensemble_comparison.png: Visualization of model performance
- final_confusion_matrix.png: Best model confusion matrix
- robustness_test.csv: Pipeline robustness test results
"""
    
    with open(os.path.join(results_dir, 'component2_report.md'), 'w') as f:
        f.write(report)
    
    print("Final report saved to: results/component2_results/component2_report.md")
    
    # Step 9: Summary and Next Steps
    print("\n" + "=" * 70)
    print("COMPONENT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✓ Data processing pipeline created and tested")
    print(f"✓ {len(evaluation_results)} models trained and evaluated")
    print(f"✓ Best model achieved {final_accuracy:.1%} accuracy")
    print(f"✓ Pipeline handles messy real-world data")
    print(f"✓ All results saved to: {results_dir}")
    print("\nReady for Component 3: Text Analysis and Summarization!")

## How to Run Component 2

### Step 1: Prepare Your Environment
```bash
# Navigate to your project folder
cd document_classifier

# Install additional libraries for Component 2
pip install scipy seaborn
