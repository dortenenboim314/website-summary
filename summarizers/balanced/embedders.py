from abc import ABC, abstractmethod
from typing import List
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    """
    
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


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI's embedding models.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
        """
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


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder using SentenceTransformer models.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize the SentenceTransformer embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
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
