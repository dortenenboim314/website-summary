from summarizers.balanced.abstractive_summarizer import AbstractiveSummarizer
from summarizers.balanced.bert_extractive_summarizer import BERTClusterSummarizer

class HybridSummarizer:
    
    def __init__(self) -> None:
        print("Initializing Hybrid Summarizer...")
        self.extractive_summarizer = BERTClusterSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer()
    
    def summarize(self, 
                 markdown: str, 
                 language: str, 
                 max_sentences: int) -> str:
        extractive_summary = self.extractive_summarizer.summarize_list(markdown, language, max_sentences)
        print(f"Extractive summary generated with {len(extractive_summary)} sentences.")
        print(f"Extractive summary: {extractive_summary}")
        max_len = max_sentences * 50
        abstractive_summary = self.abstractive_summarizer.summarize(extractive_summary, language, max_len)
        
        return abstractive_summary
        
        