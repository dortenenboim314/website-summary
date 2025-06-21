import spacy, pytextrank
from summarizers import AbstractSummarizer


# defaults
TOP_PHRASES = 3
LIMIT_PHRASES = 10
LIMIT_SENTENCES = 5


class FastSummarizer(AbstractSummarizer):
    """
    Implements a Super Lite and Fast summarization strategy.
    Prioritizes lowest latency (<300ms) using efficient extractive methods.
    """
    def __init__(self):
        self.nlps = {
            "en": spacy.load("en_core_web_sm"),
            "fr": spacy.load("fr_core_news_sm"),
            "es": spacy.load("es_core_news_sm"),
            "de": spacy.load("de_core_news_sm"),
            "zh": spacy.load("zh_core_web_sm"),
            'ar': spacy.blank('xx'),
        }
        
        for lange in self.nlps:
            self.nlps[lange].add_pipe("textrank", last=True)
    
    def summarize(self, text: str, lang: str = "en", **kwargs) -> str:
        nlp = self.nlps.get(lang)
        
        if not nlp:
            raise ValueError(f"Language '{lang}' is not supported.")

        doc = nlp(text)
        
        top_phrases = kwargs.get("top_phrases", TOP_PHRASES)
        limit_phrases = kwargs.get("limit_phrases", LIMIT_PHRASES)
        limit_sentences = kwargs.get("limit_sentences", LIMIT_SENTENCES)
        max_length_chars = kwargs.get("max_length_chars", None)

        top_phrases = [p.text for p in doc._.phrases[:top_phrases]]

        summary = [sent.text for sent in doc._.textrank.summary(limit_phrases=limit_phrases,
                                                                limit_sentences=limit_sentences,
                                                                level="sentence")]

        formatted = f"Keywords : [{', '.join(top_phrases)}]\n"
        for sent in summary:
            if max_length_chars and len(formatted) + len(sent) > max_length_chars:
                break
            formatted += f"- {sent}\n"
        return formatted

    