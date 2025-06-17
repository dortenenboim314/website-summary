from summa.summarizer import summarize
from summarizers.markdown_pre_processor import MarkdownPreprocessor
from nltk.tokenize import sent_tokenize

class GensimStyleTextRankSummarizer:
    def __init__(self):
        pass
    
    def summarize(self, markdown: str, language:str, max_sentences: int) -> str:
        text = MarkdownPreprocessor.markdown_to_clean_text(markdown)
        sentences = sent_tokenize(text)

        if len(sentences) < max_sentences:
            return text  # too short, return original or fallback

        ratio = max_sentences / len(sentences)

        try:
            return summarize(text, ratio=ratio)
        except ValueError:
            return " ".join(sentences[:max_sentences]) + "."

#main

if __name__ == "__main__":
    markdown_content = """# Machine Learning for Beginners: A Comprehensive Guide
Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It has applications in various fields such as natural language processing, computer vision, and robotics. The goal is to enable computers to learn from experience and improve their performance over time without being explicitly programmed.
Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Reinforcement learning involves training agents to make decisions by rewarding them for good actions and punishing them for bad ones.""" 
    summarizer = GensimStyleTextRankSummarizer(ratio=0.2)
    summary = summarizer.summarize(markdown_content)
    print("Summary:")
    print(summary)