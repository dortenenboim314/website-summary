from summarizers import AbstractSummarizer
from openai import OpenAI

_extractive_prompt = """
The following is the full plain text of a webpage. It may contain irrelevant sections such as headers, navigation links, author bios, newsletter banners, or footers.

{text}

---

Now, based only on the meaningful content above, select the 2-3 most important sentences that summarize the main ideas. 

- Do NOT rephrase or paraphrase—output each sentence exactly as it appears in the text.
- Return them in the order they appear.
- Do NOT add any extra text, numbering, or commentary.
"""

class ExtractiveOpenAiSummarizer(AbstractSummarizer):
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        print("OpenAI Extractive v1")
        self.client = OpenAI()
        self.model_name = model_name

    def summarize(self, text: str, lang: str, **kwargs) -> str:
        prompt = _extractive_prompt.format(text=text)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
