# from .bert_extractive_summarizer import BERTClusterSummarizer
# from .abstractive_summarizer import AbstractiveSummarizer
# from .hybrid import HybridSummarizer
# from .bart import DistilBARTSummarizer
# from .bart import BartLargeSummarizer
# from .mt5 import MT5
# from .qwen import Qwen
# from .neural_textrank import NeuralTextRank

# New imports for refactored code
from .embedders import BaseEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from .base_embedding_summarizer import BaseEmbeddingSummarizer

# Backward compatibility wrappers
from typing import Optional

class OpenAiEmbeddingSummarizer(BaseEmbeddingSummarizer):
    """
    Backward compatibility wrapper for OpenAI embedding summarizer.
    """
    def __init__(self, 
                 model_name: str = "text-embedding-3-small",
                 **kwargs):
        embedder = OpenAIEmbedder(model_name=model_name)
        super().__init__(embedder=embedder, **kwargs)

class SentenceTransformerEmbeddingSummarizer(BaseEmbeddingSummarizer):
    """
    Backward compatibility wrapper for SentenceTransformer embedding summarizer.
    """
    def __init__(self,
                 model_name: str,
                 **kwargs):
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        super().__init__(embedder=embedder, **kwargs)
