import re
import string
from typing import List, Tuple, Optional
from collections import Counter
import math

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from summarizers.markdown_pre_processor import MarkdownPreprocessor

# Language detection removed - language will be provided as parameter

class TFIDFSummarizer:
    """
    Fast TF-IDF based extractive summarizer with multilingual support.
    Optimized for low latency (20-50ms) with minimal dependencies.
    """
    
    # Supported languages for stopwords and sentence tokenization
    SUPPORTED_LANGUAGES = {
        'ar': 'arabic', 'az': 'azerbaijani', 'da': 'danish', 'de': 'german',
        'el': 'greek', 'en': 'english', 'es': 'spanish', 'fi': 'finnish',
        'fr': 'french', 'hu': 'hungarian', 'id': 'indonesian', 'it': 'italian',
        'kk': 'kazakh', 'ne': 'nepali', 'nl': 'dutch', 'no': 'norwegian',
        'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 'sl': 'slovene',
        'sv': 'swedish', 'tg': 'tajik', 'tr': 'turkish'
    }
    
    def __init__(self, 
                 position_weight: bool = False, 
                 min_sentence_length: int = 10):
        """
        Initialize TF-IDF Summarizer.
        
        Args:
            num_sentences: Maximum number of sentences in summary
            position_weight: Whether to apply position-based weighting
            min_sentence_length: Minimum character length for sentences
        """
        self.position_weight = position_weight
        self.min_sentence_length = min_sentence_length
        
        # Initialize NLTK components if available
        self._setup_nltk_components()
    
    def _setup_nltk_components(self):
        """Setup NLTK components."""
        try:
            # Download required NLTK data if not present
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    # Remove the _detect_language method entirely
    
    def _get_stopwords(self, lang_name: str) -> set:
        """Get stopwords for the specified language."""        
        try:
            return set(stopwords.words(lang_name))
        except:
            # Fallback to English stopwords
            return set(stopwords.words('english'))
    
    def _sentence_tokenize(self, text: str, lang_name: str) -> List[str]:
        """Tokenize text into sentences."""
        try:
            return sent_tokenize(text, language=lang_name)
        except:
            return sent_tokenize(text)
    
    def _word_tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except:
            # Simple fallback if word_tokenize fails
            translator = str.maketrans('', '', string.punctuation)
            clean_text = text.translate(translator).lower()
            return clean_text.split()
    
    def _preprocess_sentence(self, sentence: str, stopwords_set: set) -> List[str]:
        """Preprocess a sentence by tokenizing and removing stopwords."""
        words = self._word_tokenize(sentence)
        # Remove stopwords and short words
        return [word for word in words if word not in stopwords_set and len(word) > 2]
    
    def _calculate_tf(self, sentences: List[List[str]]) -> List[dict]:
        """Calculate Term Frequency for each sentence."""
        tf_scores = []
        
        for sentence_words in sentences:
            word_count = len(sentence_words)
            if word_count == 0:
                tf_scores.append({})
                continue
            
            word_freq = Counter(sentence_words)
            tf_dict = {word: freq / word_count for word, freq in word_freq.items()}
            tf_scores.append(tf_dict)
        
        return tf_scores
    
    def _calculate_idf(self, sentences: List[List[str]]) -> dict:
        """Calculate Inverse Document Frequency."""
        total_sentences = len(sentences)
        word_doc_count = Counter()
        
        # Count in how many sentences each word appears
        for sentence_words in sentences:
            unique_words = set(sentence_words)
            for word in unique_words:
                word_doc_count[word] += 1
        
        # Calculate IDF
        idf_dict = {}
        for word, doc_count in word_doc_count.items():
            idf_dict[word] = math.log(total_sentences / doc_count)
        
        return idf_dict
    
    def _calculate_sentence_scores(self, tf_scores: List[dict], idf_dict: dict) -> List[float]:
        """Calculate TF-IDF scores for each sentence."""
        sentence_scores = []
        
        for tf_dict in tf_scores:
            score = 0.0
            for word, tf_value in tf_dict.items():
                idf_value = idf_dict.get(word, 0)
                score += tf_value * idf_value
            sentence_scores.append(score)
        
        return sentence_scores
    
    def _apply_position_weights(self, scores: List[float]) -> List[float]:
        """Apply position-based weighting to sentence scores."""
        if not self.position_weight or len(scores) <= 2:
            return scores
        
        weighted_scores = scores.copy()
        total_sentences = len(scores)
        
        for i, score in enumerate(scores):
            # First sentence gets highest weight, last sentence gets medium weight
            if i == 0:
                weight = 1.5  # First sentence bonus
            elif i == total_sentences - 1:
                weight = 1.2  # Last sentence bonus
            elif i < total_sentences * 0.3:
                weight = 1.3  # Early sentences bonus
            else:
                weight = 1.0  # Normal weight
            
            weighted_scores[i] = score * weight
        
        return weighted_scores
    
    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Filter out sentences that are too short or not meaningful."""
        filtered = []
        for sentence in sentences:
            # Remove extra whitespace
            sentence = ' '.join(sentence.split())
            
            # Skip sentences that are too short
            if len(sentence) < self.min_sentence_length:
                continue
            
            # Skip sentences that are mostly punctuation or numbers
            alpha_chars = sum(1 for c in sentence if c.isalpha())
            if alpha_chars < len(sentence) * 0.5:
                continue
            
            filtered.append(sentence)
        
        return filtered

    def summarize(self, text: str, lang: str, **kwargs) -> str:
        """
        Generate extractive summary using TF-IDF sentence ranking.
        
        Args:
            text: Input text to summarize
            language: Language code (e.g., 'en', 'es', 'fr', 'de', etc.)
            max_sentences: Override default max_sentences
            
        Returns:
            Summary as a string
        """
        max_sentences = kwargs.get('max_sentences', 5)
        if not text or not text.strip():
            return ""

        text = MarkdownPreprocessor.markdown_to_clean_text(text)
        
        # Get language-specific stopwords
        stopwords_set = self._get_stopwords(lang)
        
        # Tokenize into sentences
        sentences = self._sentence_tokenize(text, lang)
        sentences = self._filter_sentences(sentences)
        
        # If we have fewer sentences than requested, return all
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)
        
        # Preprocess sentences (tokenize words, remove stopwords)
        processed_sentences = []
        for sentence in sentences:
            processed_words = self._preprocess_sentence(sentence, stopwords_set)
            processed_sentences.append(processed_words)
        
        # Skip sentences with no meaningful words
        valid_indices = [i for i, words in enumerate(processed_sentences) if len(words) > 0]
        if not valid_indices:
            return sentences[0] if sentences else ""
        
        # Calculate TF scores
        tf_scores = self._calculate_tf(processed_sentences)
        
        # Calculate IDF scores
        idf_dict = self._calculate_idf(processed_sentences)
        
        # Calculate TF-IDF sentence scores
        sentence_scores = self._calculate_sentence_scores(tf_scores, idf_dict)
        
        # Apply position weights
        final_scores = self._apply_position_weights(sentence_scores)
        
        # Get top sentences while maintaining original order
        scored_sentences = [(i, score) for i, score in enumerate(final_scores)]
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Sort by original position to maintain document flow
        selected_indices = sorted([idx for idx, _ in top_sentences])
        
        # Build summary
        summary_sentences = [sentences[i] for i in selected_indices]
        return ' '.join(summary_sentences)
