# Text Classification Project 🚀

A comprehensive text classification system built with Python, featuring multiple algorithms, advanced preprocessing, and a web interface.

## 🌟 Features

### Core Functionality
- **Multiple Classification Algorithms**: Logistic Regression, Random Forest, SVM, Naive Bayes, and more
- **Advanced Text Preprocessing**: Tokenization, stop word removal, lemmatization, stemming
- **Feature Extraction**: TF-IDF, Count Vectorization, LDA, LSA
- **Ensemble Methods**: Voting classifiers for improved performance
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Model Persistence**: Save and load trained models

### Advanced Features
- **Cross-validation**: Reliable performance estimation
- **Feature Importance**: Understand what drives predictions
- **Web Interface**: Flask-based API for easy deployment
- **Multiple Vectorizers**: Choose the best representation for your data
- **Customizable Preprocessing**: Tailor text cleaning to your needs

## 📁 Project Structure

```
Text-classification-/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── text_preprocessor.py      # Text preprocessing module
├── text_classifier.py        # Classification algorithms
├── main.py                  # Main demonstration script
├── app.py                   # Web interface (generated)
├── labeled_data.csv         # Sample dataset
├── trained_classifier.pkl   # Trained model (generated)
├── data_exploration.png     # Data visualizations (generated)
├── classifier_comparison.png # Model comparison (generated)
└── web_requirements.txt     # Web interface dependencies (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Demonstration

```bash
python main.py
```

This will:
- Create sample data (or use your existing `labeled_data.csv`)
- Demonstrate text preprocessing
- Train multiple classifiers
- Compare performance
- Create visualizations
- Generate a web interface

### 3. Use the Web Interface

```bash
pip install -r web_requirements.txt
python app.py
```

Open http://localhost:5000 in your browser to classify text interactively.

## 📚 Usage Examples

### Basic Text Classification

```python
from text_classifier import TextClassifier

# Initialize classifier
classifier = TextClassifier(
    vectorizer_type='tfidf',
    classifier_type='logistic_regression'
)

# Train the model
texts = ["Great product!", "Terrible service", "Okay experience"]
labels = ['positive', 'negative', 'neutral']
metrics = classifier.train(texts, labels)

# Make predictions
prediction = classifier.predict("This is amazing!")
probabilities = classifier.predict_proba("This is amazing!")

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
```

### Advanced Preprocessing

```python
from text_preprocessor import AdvancedTextPreprocessor

# Initialize preprocessor
preprocessor = AdvancedTextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    remove_numbers=True,
    remove_punctuation=True
)

# Process texts
processed_texts = preprocessor.fit_transform(texts)

# Get statistics
stats = preprocessor.get_text_statistics(texts)
vocabulary = preprocessor.get_vocabulary(texts)
```

### Ensemble Classification

```python
from text_classifier import EnsembleTextClassifier

# Initialize ensemble
ensemble = EnsembleTextClassifier(
    vectorizer_type='tfidf',
    voting_method='soft'
)

# Train ensemble
metrics = ensemble.train(texts, labels)

# Make predictions
prediction = ensemble.predict("New text to classify")
```

## 🔧 Configuration Options

### Text Preprocessing

| Parameter | Description | Options |
|-----------|-------------|---------|
| `remove_stopwords` | Remove common stop words | `True/False` |
| `lemmatize` | Apply lemmatization | `True/False` |
| `stem` | Apply stemming | `True/False` |
| `remove_numbers` | Remove numeric characters | `True/False` |
| `remove_punctuation` | Remove punctuation | `True/False` |
| `lowercase` | Convert to lowercase | `True/False` |
| `min_word_length` | Minimum word length | `int` |

### Vectorizers

| Type | Description | Use Case |
|------|-------------|----------|
| `tfidf` | Term Frequency-Inverse Document Frequency | General purpose, most common |
| `count` | Simple word count | Simple bag-of-words |
| `lda` | Latent Dirichlet Allocation | Topic modeling |
| `lsa` | Latent Semantic Analysis | Dimensionality reduction |

### Classifiers

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| `logistic_regression` | Linear classifier | Fast, interpretable | Linear decision boundary |
| `random_forest` | Ensemble of trees | Robust, handles non-linear | Slower, less interpretable |
| `svm` | Support Vector Machine | Good performance | Slower, sensitive to parameters |
| `naive_bayes` | Probabilistic classifier | Fast, works with small data | Assumes feature independence |
| `gradient_boosting` | Sequential ensemble | High performance | Can overfit, slower |

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC AUC**: Area under ROC curve (binary classification)

## 🎯 Use Cases

### 1. Sentiment Analysis
- Product reviews
- Social media posts
- Customer feedback
- Survey responses

### 2. Content Classification
- News articles
- Blog posts
- Support tickets
- Email categorization

### 3. Spam Detection
- Email filtering
- Comment moderation
- Message classification

### 4. Topic Classification
- Document categorization
- Content tagging
- Knowledge organization

## 🔍 Data Requirements

### Input Format
Your data should be in CSV format with at least two columns:
- `text`: The text content to classify
- `label`: The target classification label

### Example Dataset
```csv
text,label
"This product is amazing!",positive
"Terrible service, very disappointed.",negative
"The quality is okay, nothing special.",neutral
```

### Data Quality Tips
- **Balanced Classes**: Ensure each label has sufficient examples
- **Text Length**: Vary text lengths for robust training
- **Quality Labels**: Use consistent, accurate labeling
- **Sufficient Data**: More data generally leads to better performance

## 🚀 Advanced Usage

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# Perform tuning
tuning_results = classifier.hyperparameter_tuning(
    texts, labels, param_grid, cv=3
)

print(f"Best parameters: {tuning_results['best_params']}")
print(f"Best score: {tuning_results['best_score']}")
```

### Cross-Validation

```python
# Perform cross-validation
cv_scores = classifier.cross_validate(texts, labels, cv=5)

print(f"CV Accuracy: {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f}")
print(f"CV F1 Score: {cv_scores['f1_macro_mean']:.4f} ± {cv_scores['f1_macro_std']:.4f}")
```

### Model Persistence

```python
# Save model
classifier.save_model('my_model.pkl')

# Load model
classifier.load_model('my_model.pkl')
```

## 🌐 Web Interface

The generated web interface provides:

- **Simple UI**: Easy-to-use text input and classification
- **Real-time Results**: Instant classification with confidence scores
- **Probability Display**: See confidence for all possible labels
- **RESTful API**: Programmatic access to the classifier

### API Endpoints

- `GET /`: Web interface
- `POST /classify`: Classification endpoint

### API Usage

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is fantastic!"}'
```

## 🔧 Customization

### Adding New Classifiers

```python
from sklearn.linear_model import RidgeClassifier

class CustomTextClassifier(TextClassifier):
    def _get_classifier(self):
        if self.classifier_type == 'ridge':
            return RidgeClassifier()
        else:
            return super()._get_classifier()
```

### Custom Preprocessing

```python
class CustomPreprocessor(TextPreprocessor):
    def custom_clean(self, text):
        # Add your custom cleaning logic
        text = text.replace('custom_pattern', 'replacement')
        return super().clean_text(text)
```

## 📈 Performance Optimization

### Tips for Better Results

1. **Data Quality**: Clean, consistent, and well-labeled data
2. **Feature Engineering**: Experiment with different vectorizers
3. **Hyperparameter Tuning**: Use cross-validation to find optimal parameters
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Regularization**: Prevent overfitting with appropriate regularization

### Scaling Considerations

- **Small Datasets** (< 1000 samples): Use simple models like Naive Bayes
- **Medium Datasets** (1000-10000 samples): Logistic Regression, SVM
- **Large Datasets** (> 10000 samples): Random Forest, Gradient Boosting

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_features` in vectorizer
2. **Slow Training**: Use smaller parameter grids or fewer CV folds
3. **Poor Performance**: Check data quality and balance
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed information about the training process
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional preprocessing techniques
- New classification algorithms
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with scikit-learn, NLTK, and other open-source libraries
- Inspired by modern text classification best practices
- Designed for educational and production use

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section
2. Review the code examples
3. Open an issue on the project repository
4. Consult the scikit-learn and NLTK documentation

---

**Happy Text Classifying! 🎉**

