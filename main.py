#!/usr/bin/env python3
"""
Main script for Text Classification Project

This script demonstrates how to use the text classification system
with your own data or sample data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from text_preprocessor import TextPreprocessor, AdvancedTextPreprocessor
from text_classifier import TextClassifier, EnsembleTextClassifier, compare_classifiers


def load_or_create_sample_data():
    """
    Load data from CSV file or create sample data if file doesn't exist.
    """
    try:
        # Try to load existing data
        df = pd.read_csv('labeled_data.csv')
        print(f"Loaded existing dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("No existing dataset found. Creating sample data...")
        
        # Create sample dataset for demonstration
        sample_texts = [
            # Positive reviews
            "This product exceeded all my expectations! The quality is outstanding.",
            "Amazing experience, highly recommend to everyone.",
            "Excellent customer service and fast delivery.",
            "Fantastic product, worth every penny.",
            "Outstanding quality and great value for money.",
            "Wonderful experience, will definitely buy again.",
            "Superb product with excellent features.",
            "Brilliant design and superior performance.",
            "Exceptional service and top-notch quality.",
            "Outstanding work, very professional team.",
            
            # Negative reviews
            "Terrible service, very disappointed with the quality.",
            "Poor quality, waste of money.",
            "Not worth the price at all.",
            "Disappointing performance and bad customer support.",
            "Complete failure, avoid this company.",
            "Awful experience, would not recommend.",
            "Horrible quality and terrible service.",
            "Waste of time and money.",
            "Very poor performance, avoid at all costs.",
            "Terrible customer support and bad product.",
            
            # Neutral reviews
            "The service was okay, nothing special.",
            "Product is decent, could be better.",
            "Average quality, meets basic expectations.",
            "Service is acceptable, room for improvement.",
            "Product works as described, nothing extraordinary.",
            "Decent experience, neither good nor bad.",
            "Average performance, meets minimum requirements.",
            "Service is adequate, nothing to complain about.",
            "Product is functional, but not impressive.",
            "Okay experience, nothing remarkable."
        ]
        
        sample_labels = (
            ['positive'] * 10 + 
            ['negative'] * 10 + 
            ['neutral'] * 10
        )
        
        df = pd.DataFrame({
            'text': sample_texts,
            'label': sample_labels
        })
        
        # Save sample data
        df.to_csv('labeled_data.csv', index=False)
        print(f"Created sample dataset: {df.shape}")
        
        return df


def explore_data(df):
    """
    Explore and visualize the dataset.
    """
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Label distribution
    print("\n📈 Label Distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    
    # Text statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n📏 Text Statistics:")
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    print(f"Total vocabulary size: {len(set(' '.join(df['text']).split()))} unique words")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Label distribution pie chart
    axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Label Distribution')
    
    # Label distribution bar chart
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=axes[0, 1])
    axes[0, 1].set_title('Label Counts')
    axes[0, 1].set_xlabel('Labels')
    axes[0, 1].set_ylabel('Count')
    
    # Text length distribution
    axes[1, 0].hist(df['text_length'], bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Text Length Distribution')
    axes[1, 0].set_xlabel('Character Count')
    axes[1, 0].set_ylabel('Frequency')
    
    # Word count distribution
    axes[1, 1].hist(df['word_count'], bins=15, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Word Count Distribution')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


def demonstrate_preprocessing(df):
    """
    Demonstrate text preprocessing capabilities.
    """
    print("\n" + "="*60)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Show original texts
    print("\nOriginal texts:")
    for i, text in enumerate(df['text'].head(5), 1):
        print(f"{i}. {text}")
    
    # Basic preprocessing
    basic_preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        remove_numbers=True,
        remove_punctuation=True
    )
    
    # Advanced preprocessing with emoji removal
    advanced_preprocessor = AdvancedTextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        remove_numbers=True,
        remove_punctuation=True
    )
    
    # Process texts
    basic_processed = basic_preprocessor.fit_transform(df['text'].head(5))
    advanced_processed = advanced_preprocessor.fit_transform(df['text'].head(5))
    
    print("\nAdvanced preprocessing results (with emoji removal):")
    for i, (original, processed) in enumerate(zip(df['text'].head(5), advanced_processed), 1):
        print(f"{i}. Original: {original}")
        print(f"   Processed: {processed}")
        print()
    
    # Compare vocabulary sizes
    print(f"Vocabulary comparison:")
    print(f"  Basic preprocessing: {len(basic_preprocessor.get_vocabulary(df['text']))} words")
    print(f"  Advanced preprocessing: {len(advanced_preprocessor.get_vocabulary(df['text']))} words")
    
    # Get text statistics
    basic_stats = basic_preprocessor.get_text_statistics(df['text'])
    print(f"\n📈 Text statistics (basic preprocessing):")
    for key, value in basic_stats.items():
        print(f"  {key}: {value}")
    
    return basic_preprocessor, advanced_preprocessor


def train_single_classifier(df, preprocessor):
    """
    Train a single text classifier and demonstrate its capabilities.
    """
    print("\n" + "="*60)
    print("TRAINING CLASSIFIER")
    print("="*60)
    
    # Initialize classifier
    classifier = TextClassifier(
        vectorizer_type='tfidf',
        classifier_type='logistic_regression',
        random_state=42
    )
    
    # Train the model
    print("Training classifier...")
    metrics = classifier.train(df['text'], df['label'], test_size=0.3)
    
    print(f"\nTraining completed!")
    print(f"Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Make predictions
    print(f"\nMaking predictions on new examples:")
    new_examples = [
        "This product is absolutely fantastic!",
        "Terrible experience, very poor quality.",
        "The service was decent, could be better.",
        "Amazing results, exceeded all expectations!"
    ]
    
    for example in new_examples:
        prediction = classifier.predict(example)
        probabilities = classifier.predict_proba(example)
        
        print(f"\nText: '{example}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probabilities.values()):.3f}")
        print(f"All probabilities: {probabilities}")
    
    # Get detailed evaluation
    evaluation = classifier.evaluate()
    print(f"\n📋 Detailed evaluation:")
    print(f"Classification Report:")
    print(evaluation['classification_report'])
    
    # Save the model
    classifier.save_model('trained_classifier.pkl')
    
    return classifier


def train_ensemble_classifier(df, preprocessor):
    """
    Train an ensemble classifier and compare with single classifier.
    """
    print("\n" + "="*60)
    print("ENSEMBLE CLASSIFIER TRAINING")
    print("="*60)
    
    # Initialize ensemble classifier
    ensemble_classifier = EnsembleTextClassifier(
        vectorizer_type='tfidf',
        classifiers=['logistic_regression', 'random_forest', 'svm'],
        random_state=42
    )
    
    # Train ensemble
    print("Training ensemble classifier...")
    ensemble_metrics = ensemble_classifier.train(df['text'], df['label'], test_size=0.3)
    
    print(f"\nEnsemble training completed!")
    print(f"Performance metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Make predictions
    print(f"\n🔮 Ensemble predictions on new examples:")
    test_examples = [
        "This is absolutely fantastic!",
        "Terrible experience, very poor quality.",
        "The product is okay, nothing special."
    ]
    
    for example in test_examples:
        prediction = ensemble_classifier.predict(example)
        print(f"\nText: '{example}'")
        print(f"Prediction: {prediction}")
    
    return ensemble_classifier


def compare_all_classifiers(df):
    """
    Compare multiple classifiers on the same dataset.
    """
    print("\n" + "="*60)
    print("⚖️ CLASSIFIER COMPARISON")
    print("="*60)
    
    # Prepare data
    texts = df['text']
    labels = df['label']
    
    print("Comparing multiple classifiers...")
    comparison_results = compare_classifiers(texts, labels, test_size=0.2, random_state=42)
    
    # Display results
    print(f"\nComparison Results:")
    print("-" * 50)
    
    results_data = []
    for classifier_name, result in comparison_results.items():
        if result:
            results_data.append({
                'Classifier': classifier_name.replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'F1 Score (Weighted)': result['f1_weighted'],
                'Precision (Weighted)': result['precision_weighted'],
                'Recall (Weighted)': result['recall_weighted']
            })
            print(f"{classifier_name.replace('_', ' ').title():<20} | "
                  f"F1: {result['f1_weighted']:.4f} | "
                  f"Accuracy: {result['accuracy']:.4f}")
        else:
            print(f"{classifier_name.replace('_', ' ').title():<20} | Failed")
    
    # Create comparison visualization
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Classifier', y='Accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        sns.barplot(data=results_df, x='Classifier', y='F1 Score (Weighted)', ax=axes[0, 1])
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score (Weighted)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        sns.barplot(data=results_df, x='Classifier', y='Precision (Weighted)', ax=axes[1, 0])
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].set_ylabel('Precision (Weighted)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        sns.barplot(data=results_df, x='Classifier', y='Recall (Weighted)', ax=axes[1, 1])
        axes[1, 1].set_title('Recall Comparison')
        axes[1, 1].set_ylabel('Recall (Weighted)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return comparison_results


def demonstrate_hyperparameter_tuning(df):
    """
    Demonstrate hyperparameter tuning capabilities.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("="*60)
    
    # Prepare data
    texts = df['text']
    labels = df['label']
    
    # Initialize classifier
    classifier = TextClassifier(
        vectorizer_type='tfidf',
        classifier_type='logistic_regression',
        random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
    
    print("Performing hyperparameter tuning...")
    print(f"Parameter grid: {param_grid}")
    
    try:
        tuning_results = classifier.hyperparameter_tuning(
            texts, labels, param_grid, cv=3
        )
        
        print(f"\nHyperparameter tuning completed!")
        print(f"Best parameters: {tuning_results['best_params']}")
        print(f"Best cross-validation score: {tuning_results['best_score']:.4f}")
        
        # Retrain with best parameters
        print("\nRetraining with best parameters...")
        final_metrics = classifier.train(texts, labels, test_size=0.3)
        
        print(f"Final performance metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return classifier
        
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        return None


def create_web_interface_demo():
    """
    Create a simple demonstration of how to use the trained model.
    """
    print("\n" + "="*60)
    print("🌐 WEB INTERFACE DEMONSTRATION")
    print("="*60)
    
    # Create a simple Flask app template
    flask_template = '''
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    model_data = joblib.load('trained_classifier.pkl')
    classifier = model_data['pipeline']
    vectorizer = model_data['vectorizer']
    label_encoder = model_data['label_encoder']
    preprocessor = model_data['preprocessor']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/')
def home():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Text Classification API</h1>
            <p>Enter text below to classify it:</p>
            <textarea id="textInput" placeholder="Enter your text here..."></textarea>
            <br>
            <button onclick="classifyText()">Classify Text</button>
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
        async function classifyText() {
            const text = document.getElementById('textInput').value;
            if (!text) {
                alert('Please enter some text!');
                return;
            }
            
            const response = await fetch('/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            
            const result = await response.json();
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <h3>Classification Result:</h3>
                <p><strong>Text:</strong> ${text}</p>
                <p><strong>Predicted Label:</strong> ${result.prediction}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                <h4>All Probabilities:</h4>
                <ul>
                    ${Object.entries(result.probabilities).map(([label, prob]) => 
                        `<li><strong>${label}:</strong> ${(prob * 100).toFixed(1)}%</li>`
                    ).join('')}
                </ul>
            `;
            resultDiv.style.display = 'block';
        }
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/classify', methods=['POST'])
def classify():
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        processed_text = preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        prediction = classifier.predict(text_vectorized)[0]
        probabilities = classifier.predict_proba(text_vectorized)[0]
        
        # Convert back to original labels
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Create probabilities dictionary
        prob_dict = dict(zip(label_encoder.classes_, probabilities))
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': float(confidence),
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    # Save the Flask app
    with open('app.py', 'w') as f:
        f.write(flask_template)
    
    print("✅ Created Flask web application (app.py)")
    print("🚀 To run the web interface:")
    print("   1. Install Flask: pip install flask")
    print("   2. Run: python app.py")
    print("   3. Open: http://localhost:5000")
    
    # Create requirements file for web interface
    web_requirements = '''flask>=2.0.0
joblib>=1.1.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
'''
    
    with open('web_requirements.txt', 'w') as f:
        f.write(web_requirements)
    
    print("✅ Created web requirements file (web_requirements.txt)")


def main():
    """
    Main function to run the complete text classification demonstration.
    """
    print("🚀 TEXT CLASSIFICATION PROJECT DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Load or create data
    df = load_or_create_sample_data()
    
    # Step 2: Explore data
    df = explore_data(df)
    
    # Step 3: Demonstrate preprocessing
    basic_preprocessor, advanced_preprocessor = demonstrate_preprocessing(df)
    
    # Step 4: Train single classifier
    classifier = train_single_classifier(df, basic_preprocessor)
    
    # Step 5: Train ensemble classifier
    ensemble = train_ensemble_classifier(df, basic_preprocessor)
    
    # Step 6: Compare all classifiers
    comparison_results = compare_all_classifiers(df)
    
    # Step 7: Demonstrate hyperparameter tuning
    tuned_classifier = demonstrate_hyperparameter_tuning(df)
    
    # Step 8: Create web interface demo
    create_web_interface_demo()
    
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETED!")
    print("="*60)
    print("\n📁 Files created:")
    print("  - labeled_data.csv (your dataset)")
    print("  - trained_classifier.pkl (trained model)")
    print("  - data_exploration.png (data visualizations)")
    print("  - classifier_comparison.png (model comparison)")
    print("  - app.py (web interface)")
    print("  - web_requirements.txt (web dependencies)")
    
    print("\n🚀 Next steps:")
    print("  1. Replace sample data with your own labeled dataset")
    print("  2. Experiment with different preprocessing options")
    print("  3. Try different classifiers and vectorizers")
    print("  4. Deploy the web interface")
    print("  5. Integrate with your applications")
    
    print("\n💡 Tips:")
    print("  - Use more data for better performance")
    print("  - Experiment with different text preprocessing techniques")
    print("  - Try ensemble methods for improved accuracy")
    print("  - Use cross-validation for reliable performance estimates")
    print("  - Save and reuse trained models")


if __name__ == "__main__":
    main()
