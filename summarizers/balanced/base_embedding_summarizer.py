from typing import Optional, List
from summarizers import AbstractSummarizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import math
from .embedders import BaseEmbedder


class BaseEmbeddingSummarizer(AbstractSummarizer):
    """
    Base class for embedding-based summarizers.
    Provides common functionality for summarizers that use embeddings
    to calculate sentence similarity with the document.
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize the base embedding summarizer.
        
        Args:
            embedder: The embedder instance to use for generating embeddings
            score_normalization: Method to normalize similarity scores.
                Options: None, 'word_count', 'sqrt', 'log'
        """
        self.embedder = embedder
        
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
        
        should_use_cached_embeddings = kwargs.get("use_cached_embeddings", True)
        
        
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
        if should_use_cached_embeddings:
            embeddings = self.embedder.embed_texts_cached(to_embed)
        else:
            embeddings = self.embedder.embed_texts(to_embed)
        
        sentence_embeddings = embeddings[:-1]
        doc_embedding = embeddings[-1].reshape(1, -1)
        
        # Compute cosine similarity
        sims = cosine_similarity(sentence_embeddings, doc_embedding).flatten()
        
        # Get indices of top scoring sentences
        top_indices = np.argsort(sims)[-top_sentences:][::-1]
        
        # Return them in the order they appeared in the text
        top_indices_sorted = sorted(top_indices)
        selected_sentences = [sentences[i] for i in top_indices_sorted]
        
        return "\n".join(selected_sentences).strip() if selected_sentences else ""
