from summarizers import AbstractSummarizer, MarkdownPreprocessor

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class NeuralTextRank(AbstractSummarizer):
    def __init__(self, model_name: str='paraphrase-multilingual-mpnet-base-v2'):
        print("Neural TextRank v3!")
        # Initialize SentenceTransformer model (MiniLM)
        self.model = SentenceTransformer(model_name)
        

    def summarize(self, text: str, lang: str, **kwargs) -> str:
        # Preprocess text
        num_sentences = kwargs.get("num_sentences", 3)
        damping_factor = kwargs.get("damping_factor", 0.85)
        max_iter = kwargs.get("max_iter", 100)
        
        text = MarkdownPreprocessor.markdown_to_clean_text(text)
        
        # sentences= list(set(nltk.sent_tokenize(text, language=lang)))
        parser = PlaintextParser.from_string(text, Tokenizer(language=lang))
        sentences = list(set([x._text for x in list(parser.document.sentences)]))
        
        print(f"Number of sentences: {len(sentences)}")
        embeddings = self.model.encode(sentences, convert_to_tensor=False)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Build graph using similarity matrix
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank to rank sentences
        scores = nx.pagerank(graph, alpha=damping_factor, max_iter=max_iter)
        
        # Sort sentences by score and select top ones
        ranked_sentences = sorted(
            [(scores[i], sentence) for i, sentence in enumerate(sentences)],
            reverse=True
        )

        # Return top num_sentences, preserving original order
        selected = [s[1] for s in ranked_sentences[:num_sentences]]
        selected = sorted(selected, key=lambda s: sentences.index(s))
        return '\n'.join(selected)
