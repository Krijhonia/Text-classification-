#!/usr/bin/env python3
"""
Quick Start Script for Text Classification

This script provides a simple example to get you started with text classification.
Run this first to see the system in action!
"""

import pandas as pd
import numpy as np
from text_preprocessor import TextPreprocessor
from text_classifier import TextClassifier

def quick_demo():
    """Quick demonstration of text classification."""
    
    print("🚀 QUICK START TEXT CLASSIFICATION DEMO")
    print("=" * 50)
    
    # 1. Create simple sample data
    print("\n1️⃣ Creating sample data...")
    texts = [
        "I love this product! It's amazing!",
        "This is terrible, very disappointed.",
        "The quality is okay, nothing special.",
        "Outstanding service and great results!",
        "Poor performance, waste of money.",
        "Excellent experience, highly recommend!",
        "Bad customer support, avoid this company.",
        "Fantastic product, worth every penny!",
        "Disappointing quality, not recommended.",
        "Wonderful service, exceeded expectations!"
    ]
    
    labels = ['positive', 'negative', 'neutral', 'positive', 'negative', 
              'positive', 'negative', 'positive', 'negative', 'positive']
    
    print(f"✅ Created {len(texts)} sample texts")
    
    # 2. Initialize preprocessor
    print("\n2️⃣ Setting up text preprocessing...")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        remove_numbers=True,
        remove_punctuation=True
    )
    
    # 3. Preprocess texts
    print("3️⃣ Preprocessing texts...")
    processed_texts = preprocessor.fit_transform(texts)
    
    print("\n📝 Preprocessing example:")
    for i, (original, processed) in enumerate(zip(texts[:3], processed_texts[:3]), 1):
        print(f"Original {i}: {original}")
        print(f"Processed {i}: {processed}")
        print()
    
    # 4. Initialize and train classifier
    print("4️⃣ Training text classifier...")
    classifier = TextClassifier(
        vectorizer_type='tfidf',
        classifier_type='logistic_regression',
        random_state=42
    )
    
    # Train the model
    metrics = classifier.train(texts, labels, test_size=0.3)
    
    print(f"✅ Training completed!")
    print(f"📊 Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 5. Make predictions
    print("\n5️⃣ Making predictions on new examples...")
    new_examples = [
        "This product is absolutely fantastic!",
        "Terrible experience, very poor quality.",
        "The service was decent, could be better.",
        "Amazing results, exceeded all expectations!"
    ]
    
    print("\n🔮 Predictions:")
    for example in new_examples:
        prediction = classifier.predict(example)
        probabilities = classifier.predict_proba(example)
        confidence = max(probabilities.values())
        
        print(f"\nText: '{example}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All probabilities: {probabilities}")
    
    # 6. Save the model
    print("\n6️⃣ Saving the trained model...")
    classifier.save_model('quick_start_model.pkl')
    print("✅ Model saved as 'quick_start_model.pkl'")
    
    print("\n" + "=" * 50)
    print("🎉 QUICK DEMO COMPLETED!")
    print("=" * 50)
    print("\n💡 What you learned:")
    print("  ✓ How to preprocess text data")
    print("  ✓ How to train a text classifier")
    print("  ✓ How to make predictions")
    print("  ✓ How to save trained models")
    
    print("\n🚀 Next steps:")
    print("  1. Run 'python main.py' for the full demonstration")
    print("  2. Replace sample data with your own dataset")
    print("  3. Experiment with different classifiers")
    print("  4. Try the web interface")
    
    return classifier

def interactive_classification(classifier):
    """Interactive text classification."""
    print("\n" + "=" * 50)
    print("🎯 INTERACTIVE CLASSIFICATION")
    print("=" * 50)
    print("Enter text to classify (or 'quit' to exit):")
    
    while True:
        user_input = input("\n📝 Enter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            print("❌ Please enter some text!")
            continue
        
        try:
            # Make prediction
            prediction = classifier.predict(user_input)
            probabilities = classifier.predict_proba(user_input)
            confidence = max(probabilities.values())
            
            print(f"\n🔮 Classification Result:")
            print(f"  Text: {user_input}")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  All probabilities:")
            
            for label, prob in probabilities.items():
                print(f"    {label}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        # Run the quick demo
        classifier = quick_demo()
        
        # Ask if user wants interactive mode
        response = input("\n🤔 Would you like to try interactive classification? (y/n): ").strip().lower()
        
        if response in ['y', 'yes', 'yep', 'sure']:
            interactive_classification(classifier)
        else:
            print("👋 Thanks for trying the demo!")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure you have installed all dependencies:")
        print("     pip install -r requirements.txt")
        print("  2. Check that all files are in the same directory")
        print("  3. Ensure you have Python 3.7+ installed")
