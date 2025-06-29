from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
from pathlib import Path
import os


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders with caching support.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base embedder with caching support.
        
        Args:
            model_name: Name of the model used for embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(f".embeddings/{model_name}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk if available."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for the given text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def embed_texts_cached(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with caching support.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        # Separate texts into cached and uncached
        text_hashes = [self._get_text_hash(text) for text in texts]
        uncached_indices = []
        uncached_texts = []
        
        # Check which texts need to be embedded
        for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
            if text_hash not in self.cache:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Embed uncached texts if any
        if uncached_texts:
            new_embeddings = self.embed_texts(uncached_texts)
            
            # Update cache with new embeddings
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                text_hash = text_hashes[idx]
                self.cache[text_hash] = new_embeddings[i]
            
            # Save updated cache
            self._save_cache()
        
        # Construct result array
        embeddings_list = []
        for text_hash in text_hashes:
            embeddings_list.append(self.cache[text_hash])
        
        return np.array(embeddings_list)
    
    def clear_cache(self):
        """Clear the cache both in memory and on disk."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
    
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
        super().__init__(model_name)
        self.client = OpenAI()
    
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
        super().__init__(model_name)
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
