from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ModelEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        self.results = {}
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate a single model's performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
        """
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        # Print results
        print(f"\n{model_name} Model Results:")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("\nDetailed Report:")
        print(classification_report(y_true, y_pred))
    
    def compare_models(self):
        """
        Compare all evaluated models
        """
        if not self.results:
            print("No models evaluated yet!")
            return
        
        print("\nModel Comparison:")
        print("-" * 50)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['report']['weighted avg']['precision'],
                'Recall': results['report']['weighted avg']['recall'],
                'F1-Score': results['report']['weighted avg']['f1-score']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, categories):
        """
        Plot confusion matrix for a model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            categories (list): List of category names
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        
        # Save plot
        plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_df):
        """
        Plot model comparison chart
        
        Args:
            comparison_df (pandas.DataFrame): Comparison results
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot accuracy
        ax1.bar(comparison_df['Model'], comparison_df['Accuracy'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Plot precision
        ax2.bar(comparison_df['Model'], comparison_df['Precision'])
        ax2.set_title('Model Precision Comparison')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        
        # Plot recall
        ax3.bar(comparison_df['Model'], comparison_df['Recall'])
        ax3.set_title('Model Recall Comparison')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        
        # Plot F1-score
        ax4.bar(comparison_df['Model'], comparison_df['F1-Score'])
        ax4.set_title('Model F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    print("Model Evaluator module loaded successfully!")
