from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

class SummarizationScore(BaseModel):
    """Structured output for summarization evaluation scores"""
    faithfulness: int = Field(..., ge=1, le=5, description="How accurately does the summary reflect the source content? (1=completely inaccurate, 5=perfectly accurate)")
    relevance: int = Field(..., ge=1, le=5, description="How well does the summary capture the most important information? (1=misses key points, 5=captures all important points)")
    coherence: int = Field(..., ge=1, le=5, description="How well-structured and readable is the summary? (1=confusing/disjointed, 5=clear and well-organized)")
    conciseness: int = Field(..., ge=1, le=5, description="How appropriately condensed is the summary? (1=too verbose or too brief, 5=perfect length)")
    language_consistency: bool = Field(..., description="Is the summary in the same language as the original?")
class SummarizationJudge:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano-2025-04-14"):
        """
        Initialize the GPT-based summarization judge
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: Model to use for evaluation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        self.system_prompt = """You are an expert evaluator of text summaries. Your task is to evaluate how well a summary represents plain-text content scraped from a web page.

Please evaluate the summary on these criteria:

**Faithfulness (1-5)**: How accurately does the summary reflect the source content?
- 1: Contains major factual errors or misrepresentations
- 2: Some inaccuracies or misleading statements
- 3: Mostly accurate with minor issues
- 4: Very accurate with negligible issues
- 5: Completely accurate and faithful to source

**Relevance (1-5)**: How well does the summary capture the most important information?
- 1: Misses most key points, focuses on trivial details
- 2: Captures some important points but misses several key ones
- 3: Captures main points but may miss some important details
- 4: Captures most important information with minor omissions
- 5: Perfectly identifies and includes all key information

**Coherence (1-5)**: How well-structured and readable is the summary?
- 1: Confusing, disjointed, hard to follow
- 2: Some structural issues, occasionally unclear
- 3: Generally clear but with some organizational problems
- 4: Well-structured and mostly easy to follow
- 5: Excellently organized, flows perfectly, very clear

**Conciseness (1-5)**: How appropriately condensed is the summary?
- 1: Either way too verbose or extremely brief
- 2: Somewhat too long/short for the content
- 3: Reasonable length but could be better optimized
- 4: Good length with minor length issues
- 5: Perfect balance of brevity and completeness

**Language Consistency** (boolean): Is the summary in the same language as the original?
- False: Summary is in a different language than the original
- True:  Summary is in the same language as the original

Focus on objective assessment. If the summary is in a different language than the original, it should be False regardless of content quality."""

    def evaluate_summary(self, original_markdown: str, summary: str, language: str = "english") -> SummarizationScore:
        """
        Evaluate a single summary against its original content
        
        Args:
            original_markdown: The original markdown content
            summary: The generated summary to evaluate
            language: Language of the content
            
        Returns:
            SummarizationScore object with the four scores
        """
        user_prompt = f"""**Language**: {language}

**Original Content**:
{original_markdown}

**Summary to Evaluate**:
{summary}

Please evaluate this summary against the original content using the four criteria described in the system prompt."""

        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=self.system_prompt,
                temperature=0.0,
                input=user_prompt,
                text_format=SummarizationScore
            )
            
            return response.output_parsed
            
        except Exception as e:
            print(f"Error evaluating summary: {e}")
            # Return default middle scores on error
            raise ValueError("Failed to evaluate summary") from e

# Example usage
if __name__ == "__main__":
    # Initialize judge
    judge = SummarizationJudge(model="gpt-4.1-nano-2025-04-14")
    
    # Example evaluation
    original = """# Machine Learning
    
Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data"""
    
    summary = "机器学习是人工智能的一个分支，利用算法从数据中学习。"
    
    # Single evaluation
    scores = judge.evaluate_summary(original, summary, "english")
    print(f"Scores: F={scores.faithfulness}, R={scores.relevance}, C={scores.coherence}, Con={scores.conciseness}, LC={scores.language_consistency}")
    
    # Batch evaluation example
    # df = pd.read_csv('summaries_to_evaluate.csv')
    # evaluated_df = judge.evaluate_batch(df)
    # comparison = judge.compare_methods(evaluated_df)
    # print(comparison)