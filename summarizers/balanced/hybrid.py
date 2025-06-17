from summarizers.balanced import AbstractiveSummarizer, BERTExtractiveSummarizer

class HybridSummarizer:
    
    def __init__(self) -> None:
        self.extractive_summarizer = BERTExtractiveSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer()
    
    def summarize(self, 
                 markdown: str, 
                 language: str = 'en', 
                 max_sentences: int = 3) -> str:
        
        extractive_summary = self.extractive_summarizer.summarize(markdown, language, max_sentences * 10)
        max_len = max_sentences * 50
        abstractive_summary = self.abstractive_summarizer.summarize(extractive_summary, language, max_len)
        
        return abstractive_summary
        
        