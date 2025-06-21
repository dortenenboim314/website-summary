from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as SumyTextRank
import nltk
from summarizers import MarkdownPreprocessor

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt_tab')

class SumyTextRankSummarizer:
    def __init__(self):
        self.summarizer = SumyTextRank()

    def summarize(self, text: str, lang: str = "en", **kwargs) -> str:
        text = MarkdownPreprocessor.markdown_to_clean_text(text)
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        max_sentences = kwargs.get("max_sentences", 5)
        summary = self.summarizer(parser.document, max_sentences)
        return " ".join([str(sentence) for sentence in summary])
