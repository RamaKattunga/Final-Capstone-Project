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
from src.data_loader import DataLoader
from src.text_processor import TextProcessor
from src.models import DocumentClassifier
from src.evaluator import ModelEvaluator

def main():
    """Main function to run the document classification system"""
    
    print("=" * 60)
    print("DOCUMENT CLASSIFICATION SYSTEM")
    print("=" * 60)
    
    # TODO: Create results folder if it doesn't exist
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # We need a place to save our results (charts, reports, etc.)
    # Think of this as creating a filing cabinet for our outputs
    # 
    # WHAT TO DO:
    # 1. Check if 'results' folder exists using os.path.exists()
    # 2. If it doesn't exist, create it using os.makedirs()
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 1 - Load and Process Data
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # This is like getting our research papers ready for analysis
    # We need to: load the CSV file, clean the text, split into train/test sets
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 1a. Print a status message: "1. Loading and Processing Data..."
    # 1b. Create a DataLoader instance
    # 1c. Call load_dataset() method to load the CSV file
    # 1d. Check if data loaded successfully (if data is None, print error and return)
    # 1e. Call explore_dataset() to see what data we have
    # 1f. Call split_data() to get X_train_text, X_test_text, y_train, y_test
    # 1g. Get unique categories using sorted(data['category'].unique())
    # 1h. Print the categories found
    # 1i. Create TextProcessor instance
    # 1j. Process training texts using preprocess_texts()
    # 1k. Process test texts using preprocess_texts()
    # 1l. Create TF-IDF features using create_features() with max_features=5000
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 2 - Train Classification Models
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Now we train our three different AI models to learn patterns
    # This is like teaching three different students to recognize paper types
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 2a. Print status message: "2. Training Classification Models..."
    # 2b. Create DocumentClassifier instance
    # 2c. Call train_all_models() method with training features and labels
    # 2d. Loop through model names ['linear', 'neural', 'ensemble']
    # 2e. For each successfully trained model, save it using save_model()
    # 2f. Use filepath format: f'results/{model_name}_model.pkl'
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 3 - Evaluate Model Performance  
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Time to test our trained models and see how well they perform
    # This is like giving our students a final exam
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 3a. Print status message: "3. Evaluating Model Performance..."
    # 3b. Create ModelEvaluator instance with save_dir='results/'
    # 3c. Create empty dictionary called all_results to store results
    # 3d. Create list of model_names = ['linear', 'neural', 'ensemble']
    # 3e. For each model name, check if it was successfully trained
    # 3f. If trained, get predictions using predict() method
    # 3g. If predictions successful, evaluate using evaluate_model()
    # 3h. Store results in all_results dictionary
    # 3i. Create confusion matrix using plot_confusion_matrix()
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 4 - Compare All Models and Create Final Report
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Create comprehensive comparison showing which model performed best
    # This creates the final "report card" for our AI models
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 4a. Print status message: "4. Creating Final Model Comparison..."
    # 4b. Check if all_results has any data (if all_results:)
    # 4c. Get comprehensive comparison using compare_models()
    # 4d. Create comparison visualization using plot_model_comparison()
    # 4e. Create category performance chart using plot_category_performance()
    # 4f. Generate evaluation report using generate_evaluation_report()
    # 4g. Save all results using save_all_results()
    # 4h. If no results, print "No models were successfully trained"
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 5 - Test with Sample Predictions
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Test our best model with brand new examples to see it in action
    # This shows the model working on text it has never seen before
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 5a. Print status message: "5. Testing with Sample Predictions..."
    # 5b. Check if we have any trained models (if all_results:)
    # 5c. If yes, call test_sample_predictions() function
    # 5d. If no models, print "Skipping sample predictions - no trained models available"
    # 
    # YOUR CODE HERE:
    
    
    # TODO: STEP 6 - Final Summary and Cleanup
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Print final summary showing what was accomplished
    # Tell user where to find all the output files
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 6a. Print completion banner with "=" characters
    # 6b. Print "DOCUMENT CLASSIFICATION COMPLETED!"
    # 6c. If we have results, print number of models trained
    # 6d. Find and print the best performing model using max() function
    # 6e. Print where results are saved
    # 6f. List all the file types created (PNG, TXT, CSV, PKL)
    # 6g. Print final success message
    # 
    # YOUR CODE HERE:
    

def test_sample_predictions(classifier, processor, categories):
    """
    Test the classifier with sample text to demonstrate how it works
    
    Args:
        classifier: Trained classifier
        processor: Text processor
        categories: List of categories
    """
    # TODO: Create and test sample predictions
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # This function demonstrates our classifier working on new examples
    # Like showing a trained student some new problems to solve
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 1. Create a list called sample_texts with 5 example research descriptions:
    #    - One about machine learning/AI (Technology)
    #    - One about medical/clinical research (Healthcare)  
    #    - One about financial/economic topics (Finance)
    #    - One about education/learning (Education)
    #    - One about climate/environment (Environment)
    # 
    # 2. Create expected_categories list with the correct categories for each text
    # 
    # 3. Print header: "Testing classifier with sample research paper descriptions:"
    # 
    # 4. Create variable correct_predictions = 0
    # 
    # 5. Use for loop with enumerate to go through each text and expected category:
    #    a. Print sample number and description preview (first 60 characters)
    #    b. Print expected category
    #    c. Process text using processor.preprocess_texts([text])
    #    d. Transform to features using processor.vectorizer.transform()
    #    e. Get prediction using classifier.predict('ensemble', features)
    #    f. Get probabilities using classifier.predict_proba('ensemble', features)
    #    g. Find max confidence using max(probabilities)
    #    h. Print model prediction and confidence level
    #    i. Check if prediction matches expected, increment counter if correct
    #    j. Print "CORRECT PREDICTION!" or "INCORRECT PREDICTION"
    #    k. Show detailed probability breakdown for all categories
    #    l. Print separator line
    # 
    # 6. Calculate final accuracy = correct_predictions / total_samples
    # 
    # 7. Print summary with number correct and percentage
    # 
    # 8. Print encouraging message based on performance level
    # 
    # YOUR CODE HERE:
    

if __name__ == "__main__":
    # TODO: Run the main function
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # This is the entry point - where everything starts when you run the script
    # Simply call the main() function to start the entire pipeline
    # 
    # YOUR CODE HERE:
    
    pass  # Remove this line when you add your main() function call
