from typing import Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base_embedding import BaseEmbeddingSummarizer


class SentenceTransformerEmbeddingSummarizer(BaseEmbeddingSummarizer):
    """
    Embedding-based summarizer using SentenceTransformer models.
    """
    
    def __init__(self,
                 model_name: str = "paraphrase-multilingual-mpnet-base-v2",
                 score_normalization: Optional[str] = None):
        """
        Initialize the SentenceTransformer embedding summarizer.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            score_normalization: Method to normalize similarity scores
                Options: None, 'word_count', 'sqrt', 'log'
        """
        print("SentenceTransformer Embedding Summarizer v2")
        super().__init__(score_normalization=score_normalization)
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformer.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
