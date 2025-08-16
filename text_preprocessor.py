"""
Text Preprocessing Module for Text Classification

This module provides comprehensive text preprocessing functionality including:
- Text cleaning and normalization
- Tokenization
- Stop word removal
- Lemmatization and stemming
- Feature extraction utilities
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Union, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

class TextPreprocessor:
    """
    Comprehensive text preprocessing class for text classification tasks.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 remove_numbers: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to apply lemmatization
            stem: Whether to apply stemming (overrides lemmatization)
            remove_numbers: Whether to remove numbers
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional custom stop words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize and not stem
        self.stem = stem
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        
        # POS tag patterns for better lemmatization
        self.pos_patterns = {
            'NN': 'n',  # noun
            'NNS': 'n',  # plural noun
            'NNP': 'n',  # proper noun
            'NNPS': 'n',  # plural proper noun
            'VB': 'v',  # verb
            'VBD': 'v',  # past tense verb
            'VBG': 'v',  # gerund
            'VBN': 'v',  # past participle
            'VBP': 'v',  # present tense verb
            'VBZ': 'v',  # 3rd person singular verb
            'JJ': 'a',  # adjective
            'JJR': 'a',  # comparative adjective
            'JJS': 'a',  # superlative adjective
            'RB': 'r',  # adverb
            'RBR': 'r',  # comparative adverb
            'RBS': 'r'   # superlative adverb
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from a list of tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stop words removed
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to a list of tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of stemmed tokens
        """
        if not self.stem:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to a list of tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        # Get POS tags for better lemmatization
        pos_tags = pos_tag(tokens)
        
        lemmatized_tokens = []
        for token, pos_tag in pos_tags:
            # Get the simplified POS tag
            pos = self.pos_patterns.get(pos_tag, 'n')  # Default to noun
            lemmatized_token = self.lemmatizer.lemmatize(token, pos=pos)
            lemmatized_tokens.append(lemmatized_token)
        
        return lemmatized_tokens
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of filtered tokens
        """
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stop words
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Apply stemming or lemmatization
        if self.stem:
            tokens = self.apply_stemming(tokens)
        elif self.lemmatize:
            tokens = self.apply_lemmatization(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Apply preprocessing to a list of texts.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            List of preprocessed text strings
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        return [self.preprocess_text(text) for text in texts]
    
    def get_vocabulary(self, texts: Union[List[str], pd.Series]) -> set:
        """
        Get vocabulary from a list of texts.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            Set of unique words
        """
        processed_texts = self.fit_transform(texts)
        vocabulary = set()
        
        for text in processed_texts:
            vocabulary.update(text.split())
        
        return vocabulary
    
    def get_text_statistics(self, texts: Union[List[str], pd.Series]) -> dict:
        """
        Get statistics about the text data.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            Dictionary with text statistics
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        stats = {
            'total_texts': len(texts),
            'total_characters': sum(len(str(text)) for text in texts),
            'total_words': sum(len(str(text).split()) for text in texts),
            'avg_text_length': np.mean([len(str(text)) for text in texts]),
            'avg_word_count': np.mean([len(str(text).split()) for text in texts]),
            'vocabulary_size': len(self.get_vocabulary(texts))
        }
        
        return stats


class AdvancedTextPreprocessor(TextPreprocessor):
    """
    Advanced text preprocessor with additional features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.contractions = {
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would",
            "'s": " is"  # Note: this is simplified
        }
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emojis from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with emojis removed
        """
        emoji_pattern = re.compile("["
                                  u"\\U0001F600-\\U0001F64F"  # emoticons
                                  u"\\U0001F300-\\U0001F5FF"  # symbols & pictographs
                                  u"\\U0001F680-\\U0001F6FF"  # transport & map symbols
                                  u"\\U0001F1E0-\\U0001F1FF"  # flags (iOS)
                                  u"\\U00002702-\\U000027B0"
                                  u"\\U000024C2-\\U0001F251"
                                  "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning with additional features.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Apply parent cleaning
        text = super().clean_text(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove emojis
        text = self.remove_emojis(text)
        
        return text


# Utility functions
def create_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Create n-grams from text.
    
    Args:
        text: Input text string
        n: Size of n-grams
        
    Returns:
        List of n-grams
    """
    words = text.split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)
    
    return ngrams


def extract_entities(text: str) -> List[str]:
    """
    Extract named entities from text.
    
    Args:
        text: Input text string
        
    Returns:
        List of named entities
    """
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunk_parser = RegexpParser("""
            NP: {<NNP>+}
            NP: {<NN>+}
        """)
        chunks = chunk_parser.parse(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entities.append(' '.join([token for token, pos in chunk.leaves()]))
        
        return entities
    except:
        return []


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "This is a great product! I love it so much.",
        "Terrible service, very disappointed with the quality.",
        "The customer support was okay, nothing special."
    ]
    
    print("Original texts:")
    for text in sample_texts:
        print(f"- {text}")
    
    print("\nPreprocessed texts:")
    processed_texts = preprocessor.fit_transform(sample_texts)
    for text in processed_texts:
        print(f"- {text}")
    
    print(f"\nVocabulary size: {len(preprocessor.get_vocabulary(sample_texts))}")
    print(f"Text statistics: {preprocessor.get_text_statistics(sample_texts)}")
