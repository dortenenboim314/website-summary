from typing import Optional, List
import numpy as np
from openai import OpenAI
from .base_embedding import BaseEmbeddingSummarizer

class OpenAiEmbeddingSummarizer(BaseEmbeddingSummarizer):
    """
    Embedding-based summarizer using OpenAI's embedding models.
    """
    
    def __init__(self, 
                 model_name: str = "text-embedding-3-small",
                 score_normalization: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI embedding summarizer.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            score_normalization: Method to normalize similarity scores
                Options: None, 'word_count', 'sqrt', 'log'
            **kwargs: Additional arguments (for compatibility)
        """
        print("OpenAI Embedding Summarizer v2")
        super().__init__(score_normalization=score_normalization)
        self.client = OpenAI()
        self.model_name = model_name
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        # Convert to numpy array
        embeddings = np.array([x.embedding for x in response.data])
        return embeddings
