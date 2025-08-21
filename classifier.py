from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os

class DocumentClassifier:
    def __init__(self):
        """Initialize the classifier with different models"""
        
        # Linear Regression (Logistic Regression for classification)
        self.linear_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Simple Neural Network
        self.neural_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            random_state=42,
            max_iter=500,
            alpha=0.001  # Regularization
        )
        
        # Ensemble model (combines both models)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('linear', self.linear_model),
                ('neural', self.neural_model)
            ],
            voting='soft'  # Use probabilities for voting
        )
        
        self.models = {
            'linear': self.linear_model,
            'neural': self.neural_model,
            'ensemble': self.ensemble_model
        }
    
    def train_models(self, X_train, y_train):
        """
        Train all classification models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training classification models...")
        
        for name, model in self.models.items():
            print(f"Training {name} model...")
            
            # TODO: Train the model using the fit method
            # Hint: model.fit(X_train, y_train)
            
            model.fit(X_train, y_train)
            print(f"{name} model training completed!")
    
    def predict(self, X_test, model_name='ensemble'):
        """
        Make predictions using specified model
        
        Args:
            X_test: Test features
            model_name (str): Which model to use ('linear', 'neural', 'ensemble')
            
        Returns:
            array: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        # TODO: Make predictions using the selected model
        # Hint: Use the predict method of the model
        
        predictions = self.models[model_name].predict(X_test)
        return predictions
    
    def predict_proba(self, X_test, model_name='ensemble'):
        """
        Get prediction probabilities
        
        Args:
            X_test: Test features
            model_name (str): Which model to use
            
        Returns:
            array: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        # Get prediction probabilities
        probabilities = self.models[model_name].predict_proba(X_test)
        return probabilities
    
    def save_models(self, folder_path='models/'):
        """
        Save trained models to disk
        
        Args:
            folder_path (str): Folder to save models
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for name, model in self.models.items():
            file_path = os.path.join(folder_path, f'{name}_model.pkl')
            joblib.dump(model, file_path)
            print(f"Saved {name} model to {file_path}")
    
    def load_models(self, folder_path='models/'):
        """
        Load trained models from disk
        
        Args:
            folder_path (str): Folder containing saved models
        """
        for name in self.models.keys():
            file_path = os.path.join(folder_path, f'{name}_model.pkl')
            if os.path.exists(file_path):
                self.models[name] = joblib.load(file_path)
                print(f"Loaded {name} model from {file_path}")
            else:
                print(f"Model file {file_path} not found!")

# Example usage
if __name__ == "__main__":
    # This code will run when you execute this file directly
    print("Document Classifier module loaded successfully!")
