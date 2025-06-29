from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

class SummarizationScore(BaseModel):
    """Structured output for summarization evaluation scores"""
    faithfulness: int = Field(
        ...,
        ge=1,
        le=5,
        description="How accurately does the summary reflect the source content? (1=significant errors or misrepresentations, 5=completely faithful)"
    )
    recall: int = Field(
        ...,
        ge=1,
        le=5,
        description="How well does the summary capture all important points from the source? (1=misses most key points, 5=includes all critical information)"
    )
    precision: int = Field(
        ...,
        ge=1,
        le=5,
        description="How well does the summary focus on only relevant points, avoiding irrelevant or redundant content? (1=significant irrelevant content, 5=highly concise and relevant)"
    )
    readability_clarity: int = Field(
        ...,
        ge=1,
        le=5,
        description="How clear and easy to understand is the summary? (1=confusing or unreadable, 5=clear, coherent, and engaging)"
    )
    coherence_logical_flow: int = Field(
        ...,
        ge=1,
        le=5,
        description="How well-structured and logically connected is the summary? (1=disjointed or fragmented, 5=seamlessly seamless and well-organized)"
    )
    language_consistency: bool = Field(
        ...,
        description="Is the summary in the same language as the original?"
    )


class SummarizationJudge:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini-2025-04-14"):
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
- 1: Significant factual errors or misrepresentations  
- 2: Some inaccuracies or misleading statements  
- 3: Mostly accurate with minor issues  
- 4: Very accurate with negligible issues  
- 5: Completely accurate and faithful to the source  

**Recall (1-5)**: How well does the summary capture all important points from the source? A naive summary that is identical to the input text will receive a 5, as it includes all information.  
- 1: Misses most key points, focuses on trivial details  
- 2: Captures some important points but misses several key ones  
- 3: Captures main points but may miss some important details  
- 4: Captures most important information with minor omissions  
- 5: Includes all critical information (e.g., an exact reproduction of the input text)  

**Precision (1-5)**: How well does the summary focus on only relevant points, avoiding irrelevant or redundant content? A naive summary that is identical to the input text will likely score a 1, as it includes unnecessary details.  
- 1: Significant irrelevant or redundant content  
- 2: Includes some irrelevant or redundant content  
- 3: Mostly relevant but includes minor unnecessary details  
- 4: Highly relevant with minimal irrelevant content  
- 5: Perfectly concise, including only relevant points  

**Readability/Clarity (1-5)**: How clear and easy to understand is the summary?  
- 1: Confusing, unreadable, or poorly written  
- 2: Somewhat unclear with noticeable issues in clarity  
- 3: Generally clear but with minor clarity issues  
- 4: Clear and easy to follow with negligible issues  
- 5: Exceptionally clear, coherent, and engaging  

**Coherence/Logical Flow (1-5)**: How well-structured and logically connected is the summary?  
- 1: Disjointed, fragmented, or lacks logical flow  
- 2: Some structural issues, occasionally disjointed  
- 3: Generally well-structured but with minor organizational issues  
- 4: Well-organized with smooth transitions  
- 5: Seamlessly connected, excellently organized, and flows perfectly  

**Language Consistency (Boolean)**: Is the summary in the same language as the original?  
- False: Summary is in a different language than the original  
- True: Summary is in the same language as the original

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
    judge = SummarizationJudge(model="gpt-4.1-mini-2025-04-14")
    
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
    
