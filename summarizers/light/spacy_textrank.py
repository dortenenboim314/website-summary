import spacy, pytextrank
from summarizers import AbstractSummarizer, MarkdownPreprocessor


# defaults
TOP_PHRASES = 3
LIMIT_PHRASES = 10
LIMIT_SENTENCES = 3

LANG_TO_CODE = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "chinese": "zh",
    "arabic": "ar",
}

class SpacyTextrank(AbstractSummarizer):
    """
    Implements a Super Lite and Fast summarization strategy.
    Prioritizes lowest latency (<300ms) using efficient extractive methods.
    """
    def __init__(self):
        print("fast summarizer v3")
        self.nlps = {
            "en": spacy.load("en_core_web_sm"),
            "fr": spacy.load("fr_core_news_sm"),
            "es": spacy.load("es_core_news_sm"),
            "de": spacy.load("de_core_news_sm"),
            "zh": spacy.load("zh_core_web_sm"),
            'ar': spacy.blank('xx'),
        }
        self.nlps["ar"].add_pipe("sentencizer")  # Arabic requires sentencizer
        for lang in self.nlps:
            self.nlps[lang].add_pipe("textrank", last=True)
    
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        lang = LANG_TO_CODE[lang]
        if not lang:
            raise ValueError(f"Language '{lang}' is not supported. Supported languages: {list(LANG_TO_CODE.keys())}")
        nlp = self.nlps.get(lang)
        
        if not nlp:
            raise ValueError(f"Language '{lang}' is not supported.")
        
        # text = MarkdownPreprocessor.markdown_to_clean_text(text)
        doc = nlp(text)
        
        top_phrases = kwargs.get("top_phrases", TOP_PHRASES)
        limit_phrases = kwargs.get("limit_phrases", LIMIT_PHRASES)
        limit_sentences = kwargs.get("limit_sentences", LIMIT_SENTENCES)
        max_length_chars = kwargs.get("max_length_chars", None)

        top_phrases = [p.text for p in doc._.phrases[:top_phrases]]

        summary = [sent.text for sent in doc._.textrank.summary(limit_phrases=limit_phrases,
                                                                limit_sentences=limit_sentences,
                                                                level="sentence")]

        # formatted = f"Keywords : [{', '.join(top_phrases)}]\n"
        # for sent in summary:
        #     if max_length_chars and len(formatted) + len(sent) > max_length_chars:
        #         break
        #     formatted += f"- {sent}\n"
        # return formatted
        return "\n".join(summary)

    