from functools import lru_cache
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from summarizers import MarkdownPreprocessor
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt_tab')

class SumyTextRankSummarizer:
    def __init__(self, summarizer_type:str):
        print("sumy v2")
        
        summarizer_type = summarizer_type.lower()
        
        if summarizer_type == 'text_rank':
            self.sum_cls = TextRankSummarizer
        elif summarizer_type == 'random':
            self.sum_cls = RandomSummarizer
        elif summarizer_type == 'lsa':
            self.sum_cls = LsaSummarizer
        else:
            raise ValueError(f"Unknown summarizer type: {summarizer_type}. supported types are 'text_rank', 'random', 'lsa'.")
    
        
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        
        stemmer = Stemmer(lang)
        summarizer = self.sum_cls(stemmer)
        summarizer.stop_words = get_stop_words(lang)

        max_sentences = kwargs.get("max_sentences", 3)
        summary_sentences = summarizer(parser.document, max_sentences)
        summary = '\n'.join([sentence._text for sentence in summary_sentences])
        
        return summary.strip()
        
