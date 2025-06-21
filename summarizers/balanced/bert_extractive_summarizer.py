import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from summarizers import MarkdownPreprocessor

class BERTClusterSummarizer:
    """
    BERT-based sentence embeddings with clustering for extractive summarization.
    Uses dual-model approach: specialized English model + multilingual model.
    Optimized for Balanced strategy (200-500ms latency).
    """
    
    def __init__(self, 
                 english_model_name: str = 'all-MiniLM-L6-v2',
                 multilingual_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 min_sentence_length: int = 10):
        """
        Initialize dual-model BERT summarizer.
        
        Args:
            english_model_name: Model for English text (optimized quality)
            multilingual_model_name: Model for all other languages
            min_sentence_length: Minimum character length for sentences
        """
        self.min_sentence_length = min_sentence_length
        
        print("Loading BERT models...")
        # Load both models at initialization
        self.english_model = SentenceTransformer(english_model_name)
        self.multilingual_model = SentenceTransformer(multilingual_model_name)
        print("BERT models loaded successfully!")
    
    def _get_model(self, language: str):
        """Route to appropriate model based on language."""
        if language in ['en', 'english']:
            return self.english_model
        else:
            return self.multilingual_model
    
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
    
    def _select_representative_sentences(self, 
                                       sentences: List[str], 
                                       embeddings: np.ndarray, 
                                       cluster_labels: np.ndarray,
                                       n_clusters: int) -> List[str]:
        """
        Select the sentence closest to each cluster centroid.
        
        Args:
            sentences: Original sentences
            embeddings: BERT embeddings for each sentence
            cluster_labels: Cluster assignment for each sentence
            n_clusters: Number of clusters
            
        Returns:
            List of representative sentences (one per cluster)
        """
        representative_indices = []
        
        for cluster_id in range(n_clusters):
            # Find all sentences in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get embeddings for sentences in this cluster
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate cluster centroid
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find sentence with highest cosine similarity to centroid
            similarities = cosine_similarity(cluster_embeddings, [cluster_centroid]).flatten()
            best_idx_in_cluster = np.argmax(similarities)
            original_idx = cluster_indices[best_idx_in_cluster]
            
            representative_indices.append(original_idx)
        
        # Sort by original document order to maintain flow
        representative_indices.sort()
        
        return [sentences[i] for i in representative_indices]
    
    def summarize(self, 
                markdown: str, 
                language: str = 'en', 
                max_sentences: int = 3) -> str:
        return ''.join(self.summarize_list(markdown, language, max_sentences))
    
    
    def summarize_list(self, 
                    markdown: str, 
                    language: str = 'en', 
                    max_sentences: int = 3) -> str:
        """
        Generate extractive summary using BERT embeddings and clustering.
        
        Args:
            text: Input text to summarize
            language: Language code ('en' for English, others for multilingual model)
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summary as a string
        """
        if not markdown or not markdown.strip():
            return ""
        
        markdown = MarkdownPreprocessor.markdown_to_clean_text(markdown)
        
        # Tokenize and filter sentences
        sentences = sent_tokenize(markdown)
        sentences = self._filter_sentences(sentences)
        
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)
        
        # Get appropriate model for language
        model = self._get_model(language)
        
        # Generate BERT embeddings
        embeddings = model.encode(sentences)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=max_sentences, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select representative sentences from each cluster
        representative_sentences = self._select_representative_sentences(
            sentences, embeddings, cluster_labels, max_sentences
        )
        
        return representative_sentences