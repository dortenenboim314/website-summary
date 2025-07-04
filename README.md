# Search Summaries

Extractive summarization methods for web search results, implemented in three tiers: Lite, Balanced, and High Quality.


## Project Structure

### Scripts
- `scripts/dataset_creation.py` - Creates dataset of webpage's raw_markdown using Tavily API
- `scripts/dataset_labeling.ipynb` - Add a ground truth column to the created dataset.
- `scripts/preprocess.ipynb` - Data preprocessing and cleaning
- `scripts/infer.ipynb` - Run inference with different summarization methods
- `scripts/eval.ipynb` - Evaluate summarization methods using ROUGE metrics

### Summarization Methods

#### Lite (Fast)
- `summarizers\light\sumy_summrizer.py` - TextRank and LSA implementations using Sumy library
- `summarizers\light\spacy_textrank.py` - TextRank implementation using SpaCy for sentence segmentation

#### Balanced (Best Tradeoff)
- `summarizers\balanced\base_embedding_summarizer.py` - Neural sentence embeddings with cosine similarity ranking

#### High Quality (API-based)
- `summarizers\heavy\claude.py` - Extractive summarization using Claude API

All summarizers implement the `AbstractSummarizer` interface (`summarizers\summarizer.py`):
Supports multiple languages: English, Spanish, French, German, Chinese, and Arabic.

### Additional Files
- `judge.py` - LLM-based evaluation (currently unused)
- `summarizers/balanced/deprecated/` - Experimental balanced methods
- `summarizers/light/deprecated/` - Experimental light-weight methods




