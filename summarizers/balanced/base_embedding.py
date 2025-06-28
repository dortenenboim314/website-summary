from abc import abstractmethod
from typing import Optional, List
from summarizers import AbstractSummarizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import math


class BaseEmbeddingSummarizer(AbstractSummarizer):
    """
    Base class for embedding-based summarizers.
    Provides common functionality for summarizers that use embeddings
    to calculate sentence similarity with the document.
    """
    
    def __init__(self, score_normalization: Optional[str] = None):
        """
        Initialize the base embedding summarizer.
        
        Args:
            score_normalization: Method to normalize similarity scores.
                Options: None, 'word_count', 'sqrt', 'log'
        """
        if score_normalization not in (None, "word_count", "sqrt", "log"):
            raise ValueError(
                f"Unknown score normalization method: {score_normalization}. "
                f"Supported methods: 'word_count', 'sqrt', 'log'."
            )
        self.score_normalization = score_normalization
        
        # Eager-load spaCy models per language (high RAM), should be optimized in prod.
        self.nlps = {
            "english": spacy.load("en_core_web_sm"),
            "french": spacy.load("fr_core_news_sm"),
            "spanish": spacy.load("es_core_news_sm"),
            "german": spacy.load("de_core_news_sm"),
            "chinese": spacy.load("zh_core_web_sm"),
            "arabic": spacy.blank("xx"),
        }
        self.nlps["arabic"].add_pipe("sentencizer")
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        pass
    
    def normalize_scores(self, similarities: np.ndarray, sentences: List[str]) -> np.ndarray:
        """
        Normalize similarity scores based on sentence characteristics.
        
        Args:
            similarities: Array of similarity scores
            sentences: List of sentences
            
        Returns:
            Normalized similarity scores
        """
        if self.score_normalization == "word_count":
            lengths = [len(s.split()) for s in sentences]
            return similarities / np.array(lengths)
        
        elif self.score_normalization == "sqrt":
            lengths = [math.sqrt(len(s.split())) for s in sentences]
            return similarities / np.array(lengths)
        
        elif self.score_normalization == "log":
            lengths = [math.log(len(s.split()) + 1) for s in sentences]
            return similarities / np.array(lengths)
        else:
            return similarities
    
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        """
        Summarize text by selecting sentences with highest similarity to the document.
        
        Args:
            text: Text to summarize
            lang: Language code
            **kwargs: Additional parameters (e.g., top_sentences)
            
        Returns:
            Summary text
        """
        if lang not in self.nlps:
            raise ValueError(
                f"Language '{lang}' is not supported. "
                f"Supported languages: {list(self.nlps.keys())}"
            )
        
        nlp = self.nlps[lang]
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        top_sentences = kwargs.get("top_sentences", 3)
        if len(sentences) <= top_sentences:
            return "\n".join(sentences)
        
        # Embed sentences and full document
        to_embed = sentences + [text]
        embeddings = self.embed_texts(to_embed)
        
        sentence_embeddings = embeddings[:-1]
        doc_embedding = embeddings[-1].reshape(1, -1)
        
        # Compute cosine similarity
        sims = cosine_similarity(sentence_embeddings, doc_embedding).flatten()
        
        # Apply score normalization if configured
        if self.score_normalization:
            sims = self.normalize_scores(sims, sentences)
        
        # Get indices of top scoring sentences
        top_indices = np.argsort(sims)[-top_sentences:][::-1]
        
        # Return them in the order they appeared in the text
        top_indices_sorted = sorted(top_indices)
        selected_sentences = [sentences[i] for i in top_indices_sorted]
        
        return "\n".join(selected_sentences).strip() if selected_sentences else ""
