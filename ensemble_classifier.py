from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                             BaggingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import joblib
import os

class EnsembleClassifier:
    def __init__(self):
        """Initialize ensemble classification system"""
        
        # Define base models for ensemble
        self.base_models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='linear', 
                probability=True, 
                random_state=42, 
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                alpha=0.001
            ),
            'naive_bayes': MultinomialNB(alpha=1.0)
        }
        
        # Define ensemble methods
        self.ensemble_methods = {}
        self.trained_models = {}
        
    def create_ensemble_methods(self):
        """Create different ensemble methods"""
        
        # TODO: Implement different ensemble methods
        # Hints:
        # - Voting Classifier (hard and soft voting)
        # - Bagging with different base estimators
        # - Boosting (AdaBoost)
        # - Stacking (advanced)
        
        # 1. Voting Ensemble (Soft Voting)
        voting_models = [
            ('lr', self.base_models['logistic_regression']),
            ('rf', self.base_models['random_forest']),
            ('nb', self.base_models['naive_bayes'])
        ]
        
        self.ensemble_methods['voting_soft'] = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        # 2. Voting Ensemble (Hard Voting)  
        self.ensemble_methods['voting_hard'] = VotingClassifier(
            estimators=voting_models,
            voting='hard'
        )
        
        # 3. Bagging with Random Forest
        self.ensemble_methods['bagging_rf'] = BaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=10,
            random_state=42
        )
        
        # 4. Bagging with Logistic Regression
        self.ensemble_methods['bagging_lr'] = BaggingClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_estimators=10,
            random_state=42
        )
        
        # 5. AdaBoost
        self.ensemble_methods['adaboost'] = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
        
        print(f"Created {len(self.ensemble_methods)} ensemble methods")
        return self.ensemble_methods
    
    def train_individual_models(self, X_train, y_train):
        """
        Train individual base models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training individual base models...")
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # TODO: Train the model and store it
            # Hint: Use the fit method and store in self.trained_models
            
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
    
    def train_ensemble_models(self, X_train, y_train):
        """
        Train ensemble models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training ensemble models...")
        
        # Create ensemble methods first
        self.create_ensemble_methods()
        
        for name, ensemble in self.ensemble_methods.items():
            print(f"Training {name} ensemble...")
            
            # TODO: Train the ensemble model
            # Hint: Same as individual models, use fit method
            
            try:
                ensemble.fit(X_train, y_train)
                self.trained_models[f"ensemble_{name}"] = ensemble
                print(f"✓ {name} ensemble trained successfully")
            except Exception as e:
                print(f"✗ Error training {name} ensemble: {e}")
    
    def evaluate_all_models(self, X_train, y_train, cv_folds=5):
        """
        Evaluate all models using cross-validation
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Model performance results
        """
        print(f"Evaluating all models with {cv_folds}-fold cross-validation...")
        
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # TODO: Evaluate each trained model
        # Hints:
        # - Use cross_val_score for evaluation
        # - Calculate mean and standard deviation
        # - Store results in a dictionary
        
        all_models = {**self.base_models, **self.ensemble_methods}
        
        for name, model in all_models.items():
            try:
                print(f"Evaluating {name}...")
                
                # Perform cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                
                results[name] = {
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"  Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def get_best_model(self, evaluation_results):
        """
        Identify the best performing model
        
        Args:
            evaluation_results (dict): Results from model evaluation
            
        Returns:
            tuple: (best_model_name, best_accuracy, best_model_object)
        """
        # TODO: Find the model with highest mean accuracy
        # Hint: Iterate through results and find maximum mean_accuracy
        
        best_accuracy = 0
        best_model_name = None
        
        for name, results in evaluation_results.items():
            if 'mean_accuracy' in results:
                if results['mean_accuracy'] > best_accuracy:
                    best_accuracy = results['mean_accuracy']
                    best_model_name = name
        
        # Get the actual model object
        if best_model_name in self.trained_models:
            best_model = self.trained_models[best_model_name]
        elif best_model_name in self.base_models:
            best_model = self.base_models[best_model_name]
        elif best_model_name in self.ensemble_methods:
            best_model = self.ensemble_methods[best_model_name]
        else:
            best_model = None
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.3f}")
        
        return best_model_name, best_accuracy, best_model
    
    def save_models(self, folder_path='models/ensemble_models/'):
        """
        Save all trained models
        
        Args:
            folder_path (str): Folder to save models
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save individual models
        for name, model in self.trained_models.items():
            file_path = os.path.join(folder_path, f'{name}_model.pkl')
            joblib.dump(model, file_path)
            print(f"Saved {name} to {file_path}")
    
    def predict_with_confidence(self, X_test, model_name=None):
        """
        Make predictions with confidence scores
        
        Args:
            X_test: Test features
            model_name (str): Specific model to use (if None, use best model)
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if model_name and model_name in self.trained_models:
            model = self.trained_models[model_name]
        else:
            # Use the first available ensemble model
            ensemble_models = [name for name in self.trained_models.keys() if 'ensemble' in name]
            if ensemble_models:
                model = self.trained_models[ensemble_models[0]]
            else:
                model = list(self.trained_models.values())[0]
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Get confidence scores (probabilities)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            confidence_scores = np.max(probabilities, axis=1)
        else:
            confidence_scores = np.ones(len(predictions))  # Default confidence of 1.0
        
        return predictions, confidence_scores

# Example usage
if __name__ == "__main__":
    print("Ensemble Classifier module loaded successfully!")
