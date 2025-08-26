"""
Models Module for Document Classification
Implements different machine learning models for text classification
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class DocumentClassifier:
    def __init__(self):
        """Initialize the document classifier with different models"""
        
        # TODO: Initialize different machine learning models
        # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
        # We're creating three different types of "smart classifiers" - like having three different
        # experts who each have their own way of reading and categorizing papers
        # 
        # 1. LINEAR MODEL (LogisticRegression):
        #    - Like a simple, fast reader who looks for key patterns
        #    - Good for understanding which words indicate which categories
        #    - Fast to train and usually gives decent results
        # 
        # 2. NEURAL NETWORK MODEL (MLPClassifier):
        #    - Like a more complex reader who can find hidden patterns
        #    - Has "layers" that can learn complex relationships between words
        #    - More powerful but takes longer to train
        # 
        # 3. ENSEMBLE MODEL (VotingClassifier):
        #    - Like having both experts vote on each paper
        #    - Combines the predictions of both models above
        #    - Usually gives the best results because it uses both approaches
        
        # Linear model (Logistic Regression)
        self.linear_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
        
        # Neural Network model
        self.neural_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            random_state=42,
            max_iter=500,
            alpha=0.001,  # Regularization parameter
            learning_rate_init=0.001
        )
        
        # Ensemble model (combines linear and neural network)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('linear', self.linear_model),
                ('neural', self.neural_model)
            ],
            voting='soft'  # Use probability predictions
        )
        
        # Store all models for easy access
        self.models = {
            'linear': self.linear_model,
            'neural': self.neural_model,
            'ensemble': self.ensemble_model
        }
        
        self.is_trained = {model_name: False for model_name in self.models.keys()}
        
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Args:
            model_name (str): Name of model to train ('linear', 'neural', 'ensemble')
            X_train: Training features
            y_train: Training labels
            
        Returns:
            bool: True if training successful, False otherwise
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found!")
            print(f"Available models: {list(self.models.keys())}")
            return False
        
        try:
            print(f"Training {model_name} model...")
            
            # TODO: Train the specified model
            # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
            # Training a model is like teaching someone to recognize patterns
            # Think of it like showing a child many examples of cats and dogs
            # until they can tell the difference
            # 
            # WHAT HAPPENS DURING TRAINING:
            # 1. We show the model many research papers (X_train)
            # 2. We tell it what category each paper belongs to (y_train)
            # 3. The model learns patterns: "papers with 'clinical' often = Healthcare"
            # 4. After training, it can categorize new papers it hasn't seen
            # 
            # THE FIT METHOD:
            # - model.fit(X_train, y_train) is the magic command that does the learning
            # - X_train = the text features (numbers representing words)
            # - y_train = the correct categories (Technology, Healthcare, etc.)
            
            model = self.models[model_name]
            model.fit(X_train, y_train)
            
            self.is_trained[model_name] = True
            print(f"{model_name} model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error training {model_name} model: {e}")
            return False
    
    def train_all_models(self, X_train, y_train):
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            dict: Training results for each model
        """
        print("Training all models...")
        results = {}
        
        for model_name in self.models.keys():
            success = self.train_model(model_name, X_train, y_train)
            results[model_name] = success
        
        successful_models = [name for name, success in results.items() if success]
        print(f"\nTraining Summary:")
        print(f"Successfully trained: {successful_models}")
        
        if len(successful_models) < len(self.models):
            failed_models = [name for name, success in results.items() if not success]
            print(f"Failed to train: {failed_models}")
        
        return results
    
    def predict(self, model_name, X_test):
        """
        Make predictions using a specific model
        
        Args:
            model_name (str): Name of model to use
            X_test: Test features
            
        Returns:
            array: Predictions
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found!")
            return None
        
        if not self.is_trained[model_name]:
            print(f"Error: Model '{model_name}' has not been trained yet!")
            return None
        
        try:
            # TODO: Make predictions using the specified model
            # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
            # After training, we can ask our model to predict categories for new papers
            # This is like asking a trained expert: "What category is this paper?"
            # 
            # THE PREDICTION PROCESS:
            # 1. We give the model new text features (X_test) 
            # 2. The model uses what it learned during training
            # 3. It returns its best guess for each paper's category
            # 
            # EXAMPLE:
            # Input: Text features for "Machine learning improves medical diagnosis"
            # Output: "Healthcare" (the model's prediction)
            # 
            # THE PREDICT METHOD:
            # - model.predict(X_test) returns an array of category predictions
            # - One prediction for each paper in the test set
            
            model = self.models[model_name]
            predictions = model.predict(X_test)
            return predictions
            
        except Exception as e:
            print(f"Error making predictions with {model_name}: {e}")
            return None
    
    def predict_proba(self, model_name, X_test):
        """
        Get prediction probabilities from a model
        
        Args:
            model_name (str): Name of model to use
            X_test: Test features
            
        Returns:
            array: Prediction probabilities
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found!")
            return None
        
        if not self.is_trained[model_name]:
            print(f"Error: Model '{model_name}' has not been trained yet!")
            return None
        
        try:
            model = self.models[model_name]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                return probabilities
            else:
                print(f"Model '{model_name}' does not support probability predictions")
                return None
                
        except Exception as e:
            print(f"Error getting probabilities from {model_name}: {e}")
            return None
    
    def evaluate_model(self, model_name, X_test, y_test, target_names=None):
        """
        Evaluate a specific model's performance
        
        Args:
            model_name (str): Name of model to evaluate
            X_test: Test features
            y_test: True test labels
            target_names (list): Names of target classes
            
        Returns:
            dict: Evaluation results
        """
        predictions = self.predict(model_name, X_test)
        
        if predictions is None:
            return None
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Get detailed classification report
        report = classification_report(
            y_test, predictions, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': report
        }
        
        # Print results
        print(f"\n{model_name.upper()} MODEL EVALUATION")
        print("="*50)
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
        
        return results
    
    def compare_models(self, X_test, y_test, target_names=None):
        """
        Compare all trained models
        
        Args:
            X_test: Test features
            y_test: True test labels
            target_names (list): Names of target classes
            
        Returns:
            dict: Comparison results for all models
        """
        print("\nCOMPARING ALL MODELS")
        print("="*50)
        
        comparison_results = {}
        
        for model_name in self.models.keys():
            if self.is_trained[model_name]:
                results = self.evaluate_model(model_name, X_test, y_test, target_names)
                if results:
                    comparison_results[model_name] = results
        
        # Summary comparison
        if comparison_results:
            print(f"\nMODEL COMPARISON SUMMARY")
            print("="*30)
            accuracies = {name: results['accuracy'] for name, results in comparison_results.items()}
            
            # Sort by accuracy
            sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (model_name, accuracy) in enumerate(sorted_models, 1):
                print(f"{rank}. {model_name.capitalize()}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            best_model = sorted_models[0][0]
            print(f"\nBest performing model: {best_model}")
            
            return comparison_results
        else:
            print("No trained models to compare!")
            return {}
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk
        
        Args:
            model_name (str): Name of model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models or not self.is_trained[model_name]:
            print(f"Cannot save model '{model_name}' - not trained!")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            joblib.dump(self.models[model_name], filepath)
            print(f"Model '{model_name}' saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to the saved model
        """
        try:
            loaded_model = joblib.load(filepath)
            self.models[model_name] = loaded_model
            self.is_trained[model_name] = True
            print(f"Model loaded as '{model_name}' from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Testing DocumentClassifier class...")
    
    # This would normally use real training data
    # For testing, we'll just verify the class initializes correctly
    classifier = DocumentClassifier()
    
    print("Available models:")
    for model_name in classifier.models.keys():
        print(f"  - {model_name}")
    
    print(f"\nModels initialized successfully!")
    print("Note: To fully test this class, you need training data from the previous modules.")
