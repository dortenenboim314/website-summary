from abc import ABC, abstractmethod
from typing import List, Dict, Optional
    
class AbstractSummarizer(ABC):
    """
    Abstract base class defining the interface for text summarization.
    All concrete summarizer implementations must adhere to this interface.
    """
    @abstractmethod
    def summarize(self, text: str, lang: str = "en", **kwargs) -> str:
        """
        Summarizes the given text.

        Args:
            text (str): The input text to be summarized. Can be raw markdown.
            lang (str): ISO 639-1 language code (e.g., "en", "fr", "es", "de", "ar", "zh").
                        Supports English, French, Spanish, German, Arabic, Chinese.
            **kwargs: Additional parameters specific to the summarization strategy.

        Returns:
            str: The summarized text.

        Raises:
            ValueError: If the language is not supported.
            Exception: For other summarization-specific errors.
        """
        pass