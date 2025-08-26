"""
Evaluation Module for Document Classification
Handles model evaluation, visualization, and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import os

class ModelEvaluator:
    def __init__(self, save_dir='results/'):
        """
        Initialize the model evaluator
        
        Args:
            save_dir (str): Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.evaluation_results = {}
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, target_names):
        """
        Create and save confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            target_names (list): Names of target classes
        """
        # TODO: Create confusion matrix visualization
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # A confusion matrix is like a report card that shows how well our model did
        # It's a table that compares what the model predicted vs. what was actually correct
        # 
        # HOW TO READ A CONFUSION MATRIX:
        # - Rows = True categories (what the papers actually are)
        # - Columns = Predicted categories (what our model guessed)
        # - Numbers = How many papers fell into each combination
        # 
        # EXAMPLE: If the model predicted "Technology" but the paper was actually "Healthcare",
        # that goes in the Healthcare row, Technology column
        # 
        # PERFECT MODEL: All numbers would be on the diagonal (correct predictions)
        # CONFUSED MODEL: Numbers scattered everywhere (wrong predictions)
        # 
        # STEPS TO CREATE THE VISUALIZATION:
        # 1. Calculate the confusion matrix using sklearn
        # 2. Create a heatmap (colored table) using seaborn
        # 3. Add labels so we know what each row/column represents
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Number of Predictions'}
        )
        
        plt.title(f'Confusion Matrix - {model_name.title()} Model')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        
        # Add accuracy to the plot
        accuracy = accuracy_score(y_true, y_pred)
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10)
        
        # Save the plot
        filename = f'{model_name}_confusion_matrix.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {filepath}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_results):
        """
        Create visualization comparing different models
        
        Args:
            comparison_results (dict): Results from model comparison
        """
        if not comparison_results:
            print("No comparison results to plot!")
            return
        
        # Extract accuracy scores
        model_names = list(comparison_results
