"""
Text Classification Module

This module provides comprehensive text classification functionality including:
- Multiple classification algorithms
- Feature extraction methods
- Model evaluation and comparison
- Hyperparameter tuning
- Ensemble methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from text_preprocessor import TextPreprocessor


class TextClassifier:
    """
    Comprehensive text classification class with multiple algorithms and evaluation methods.
    """
    
    def __init__(self, 
                 vectorizer_type: str = 'tfidf',
                 classifier_type: str = 'logistic_regression',
                 random_state: int = 42):
        """
        Initialize the text classifier.
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf', 'count', 'lda', 'lsa')
            classifier_type: Type of classifier
            random_state: Random state for reproducibility
        """
        self.vectorizer_type = vectorizer_type
        self.classifier_type = classifier_type
        self.random_state = random_state
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize vectorizer
        self.vectorizer = self._get_vectorizer()
        
        # Initialize classifier
        self.classifier = self._get_classifier()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Model pipeline
        self.pipeline = None
        self.is_fitted = False
        
        # Results storage
        self.results = {}
        self.feature_importance = None
    
    def _get_vectorizer(self):
        """Get the appropriate vectorizer based on type."""
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                random_state=self.random_state
            )
        elif self.vectorizer_type == 'count':
            return CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif self.vectorizer_type == 'lda':
            return LatentDirichletAllocation(
                n_components=100,
                random_state=self.random_state
            )
        elif self.vectorizer_type == 'lsa':
            return TruncatedSVD(
                n_components=100,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
    
    def _get_classifier(self):
        """Get the appropriate classifier based on type."""
        if self.classifier_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        elif self.classifier_type == 'svm':
            return SVC(random_state=self.random_state, probability=True)
        elif self.classifier_type == 'linear_svm':
            return LinearSVC(random_state=self.random_state, max_iter=1000)
        elif self.classifier_type == 'naive_bayes':
            return MultinomialNB()
        elif self.classifier_type == 'bernoulli_nb':
            return BernoulliNB()
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        elif self.classifier_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=self.random_state)
        elif self.classifier_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def preprocess_data(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Preprocess the text data.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            List of preprocessed text strings
        """
        return self.preprocessor.fit_transform(texts)
    
    def prepare_data(self, 
                    texts: Union[List[str], pd.Series], 
                    labels: Union[List[str], pd.Series],
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training and testing.
        
        Args:
            texts: List or Series of text strings
            labels: List or Series of labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, vectorizer, label_encoder)
        """
        # Preprocess texts
        processed_texts = self.preprocess_data(texts)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Fit vectorizer on training data
        if self.vectorizer_type in ['tfidf', 'count']:
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            X_test_vectorized = self.vectorizer.transform(X_test)
        else:
            # For LDA/LSA, we need to first use TF-IDF
            temp_vectorizer = TfidfVectorizer(max_features=1000)
            X_train_temp = temp_vectorizer.fit_transform(X_train)
            X_test_temp = temp_vectorizer.transform(X_test)
            
            X_train_vectorized = self.vectorizer.fit_transform(X_train_temp)
            X_test_vectorized = self.vectorizer.transform(X_test_temp)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    def train(self, 
              texts: Union[List[str], pd.Series], 
              labels: Union[List[str], pd.Series],
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the text classification model.
        
        Args:
            texts: List or Series of text strings
            labels: List or Series of labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(texts, labels, test_size)
        
        # Create and fit pipeline
        self.pipeline = Pipeline([
            ('classifier', self.classifier)
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test) if hasattr(self.classifier, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store results
        self.results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }
        
        # Get feature importance if available
        self._extract_feature_importance()
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate various performance metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract feature importance if the classifier supports it."""
        if hasattr(self.classifier, 'feature_importances_'):
            self.feature_importance = self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            self.feature_importance = np.abs(self.classifier.coef_[0])
        else:
            self.feature_importance = None
    
    def predict(self, texts: Union[List[str], str]) -> Union[List[str], str]:
        """
        Make predictions on new text data.
        
        Args:
            texts: Text or list of texts to classify
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = self.preprocess_data(texts)
        
        # Vectorize
        if self.vectorizer_type in ['tfidf', 'count']:
            X_vectorized = self.vectorizer.transform(processed_texts)
        else:
            # For LDA/LSA
            temp_vectorizer = TfidfVectorizer(max_features=1000)
            X_temp = temp_vectorizer.fit_transform(processed_texts)
            X_vectorized = self.vectorizer.transform(X_temp)
        
        # Make predictions
        predictions = self.pipeline.predict(X_vectorized)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels[0] if len(texts) == 1 else predicted_labels
    
    def predict_proba(self, texts: Union[List[str], str]) -> Union[np.ndarray, List[Dict]]:
        """
        Get prediction probabilities for new text data.
        
        Args:
            texts: Text or list of texts to classify
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError("This classifier does not support probability predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = self.preprocess_data(texts)
        
        # Vectorize
        if self.vectorizer_type in ['tfidf', 'count']:
            X_vectorized = self.vectorizer.transform(processed_texts)
        else:
            # For LDA/LSA
            temp_vectorizer = TfidfVectorizer(max_features=1000)
            X_temp = temp_vectorizer.fit_transform(processed_texts)
            X_vectorized = self.vectorizer.transform(X_temp)
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(X_vectorized)
        
        # Convert to list of dictionaries
        label_names = self.label_encoder.classes_
        result = []
        
        for prob in probabilities:
            prob_dict = dict(zip(label_names, prob))
            result.append(prob_dict)
        
        return result[0] if len(texts) == 1 else result
    
    def evaluate(self) -> Dict[str, Any]:
        """Get comprehensive evaluation results."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        return {
            'metrics': self.results['metrics'],
            'classification_report': classification_report(
                self.results['y_test'], 
                self.results['y_pred'],
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(
                self.results['y_test'], 
                self.results['y_pred']
            )
        }
    
    def cross_validate(self, 
                      texts: Union[List[str], pd.Series], 
                      labels: Union[List[str], pd.Series],
                      cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            texts: List or Series of text strings
            labels: List or Series of labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores
        """
        # Preprocess texts
        processed_texts = self.preprocess_data(texts)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Perform cross-validation
        cv_scores = {}
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        for metric in metrics:
            scores = cross_val_score(
                pipeline, processed_texts, y_encoded, 
                cv=cv, scoring=metric, n_jobs=-1
            )
            cv_scores[metric] = scores.tolist()
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        return cv_scores
    
    def hyperparameter_tuning(self, 
                             texts: Union[List[str], pd.Series], 
                             labels: Union[List[str], pd.Series],
                             param_grid: Dict[str, List],
                             cv: int = 3) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            texts: List or Series of text strings
            labels: List or Series of labels
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing tuning results
        """
        # Preprocess texts
        processed_texts = self.preprocess_data(texts)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, 
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(processed_texts, y_encoded)
        
        # Update classifier with best parameters
        self.classifier = grid_search.best_estimator_.named_steps['classifier']
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'pipeline': self.pipeline,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor,
            'vectorizer_type': self.vectorizer_type,
            'classifier_type': self.classifier_type,
            'random_state': self.random_state,
            'results': self.results,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.preprocessor = model_data['preprocessor']
        self.vectorizer_type = model_data['vectorizer_type']
        self.classifier_type = model_data['classifier_type']
        self.random_state = model_data['random_state']
        self.results = model_data['results']
        self.feature_importance = model_data['feature_importance']
        
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


class EnsembleTextClassifier:
    """
    Ensemble text classifier combining multiple models.
    """
    
    def __init__(self, 
                 vectorizer_type: str = 'tfidf',
                 voting_method: str = 'soft',
                 random_state: int = 42):
        """
        Initialize the ensemble classifier.
        
        Args:
            vectorizer_type: Type of vectorizer
            voting_method: Voting method ('hard' or 'soft')
            random_state: Random state for reproducibility
        """
        self.vectorizer_type = vectorizer_type
        self.voting_method = voting_method
        self.random_state = random_state
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize vectorizer
        self.vectorizer = self._get_vectorizer()
        
        # Initialize base classifiers
        self.base_classifiers = self._get_base_classifiers()
        
        # Initialize ensemble
        self.ensemble = VotingClassifier(
            estimators=self.base_classifiers,
            voting=voting_method
        )
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Model pipeline
        self.pipeline = None
        self.is_fitted = False
        
        # Results storage
        self.results = {}
    
    def _get_vectorizer(self):
        """Get the appropriate vectorizer."""
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                random_state=self.random_state
            )
        else:
            return CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
    
    def _get_base_classifiers(self):
        """Get base classifiers for the ensemble."""
        return [
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=self.random_state, n_estimators=100)),
            ('svm', SVC(random_state=self.random_state, probability=True)),
            ('nb', MultinomialNB())
        ]
    
    def train(self, 
              texts: Union[List[str], pd.Series], 
              labels: Union[List[str], pd.Series],
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the ensemble classifier.
        
        Args:
            texts: List or Series of text strings
            labels: List or Series of labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results
        """
        # Preprocess texts
        processed_texts = self.preprocessor.fit_transform(texts)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Fit vectorizer
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Create and fit pipeline
        self.pipeline = Pipeline([
            ('classifier', self.ensemble)
        ])
        
        # Train ensemble
        self.pipeline.fit(X_train_vectorized, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test_vectorized)
        y_pred_proba = self.pipeline.predict_proba(X_test_vectorized)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store results
        self.results = {
            'X_train': X_train_vectorized,
            'X_test': X_test_vectorized,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
    
    def predict(self, texts: Union[List[str], str]) -> Union[List[str], str]:
        """Make predictions on new text data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = self.preprocessor.fit_transform(texts)
        
        # Vectorize
        X_vectorized = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.pipeline.predict(X_vectorized)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels[0] if len(texts) == 1 else predicted_labels


def compare_classifiers(texts: Union[List[str], pd.Series], 
                       labels: Union[List[str], pd.Series],
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple classifiers on the same dataset.
    
    Args:
        texts: List or Series of text strings
        labels: List or Series of labels
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing results for each classifier
    """
    classifiers = [
        'logistic_regression',
        'random_forest',
        'svm',
        'naive_bayes',
        'gradient_boosting'
    ]
    
    results = {}
    
    for classifier_type in classifiers:
        print(f"\nTraining {classifier_type}...")
        
        try:
            classifier = TextClassifier(
                vectorizer_type='tfidf',
                classifier_type=classifier_type,
                random_state=random_state
            )
            
            metrics = classifier.train(texts, labels, test_size)
            results[classifier_type] = metrics
            
            print(f"{classifier_type} - F1 Score: {metrics['f1_weighted']:.4f}")
            
        except Exception as e:
            print(f"Error training {classifier_type}: {e}")
            results[classifier_type] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This product is amazing! I love it.",
        "Terrible quality, very disappointed.",
        "Good service, but could be better.",
        "Excellent experience, highly recommend!",
        "Poor customer support, avoid this company."
    ]
    
    sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']
    
    # Single classifier
    print("Training single classifier...")
    classifier = TextClassifier(
        vectorizer_type='tfidf',
        classifier_type='logistic_regression'
    )
    
    metrics = classifier.train(sample_texts, sample_labels)
    print(f"Training results: {metrics}")
    
    # Make predictions
    new_text = "This is a fantastic product!"
    prediction = classifier.predict(new_text)
    probabilities = classifier.predict_proba(new_text)
    
    print(f"\nPrediction for '{new_text}': {prediction}")
    print(f"Probabilities: {probabilities}")
    
    # Ensemble classifier
    print("\nTraining ensemble classifier...")
    ensemble = EnsembleTextClassifier(voting_method='soft')
    ensemble_metrics = ensemble.train(sample_texts, sample_labels)
    print(f"Ensemble results: {ensemble_metrics}")
    
    # Compare classifiers
    print("\nComparing multiple classifiers...")
    comparison_results = compare_classifiers(sample_texts, sample_labels)
    
    print("\nFinal comparison:")
    for classifier_name, result in comparison_results.items():
        if result:
            print(f"{classifier_name}: F1 = {result['f1_weighted']:.4f}")
        else:
            print(f"{classifier_name}: Failed")
