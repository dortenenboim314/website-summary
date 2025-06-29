from summarizers import AbstractSummarizer
from openai import OpenAI

class OpenAiSummarizer(AbstractSummarizer):
    def __init__(self, model_name = "gpt-4o-2024-08-06"):
        print("OpenAI v4")
        self.client = OpenAI()
        self.model_name = model_name
        
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        # print(self.system_message.format(lang=lang))
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role": "user", "content": content.format(text=text)}
            ],
            max_tokens=80
        )
        summary = response.choices[0].message.content.strip()
        return summary
        
    
content = """
The following is the full plain text of a webpage. It may contain irrelevant sections such as headers, navigation links, author bios, newsletter banners, or footers.

{text}

---

Now, based only on the meaningful content above, write a concise summary in 2-3 sentences.

- Do not include any meta language like "this article", "this guide", "this page", or "here is a summary".
- Do not mention the webpage, website, or author.
- Do not describe the structure of the content.
- Just summarize the actual content, clearly and factually.
- Do not guess or assume anything not directly stated.

IMPORTANT: The output MUST be in the same language as the input.

---

Examples:

Bad Output 1:
This article discusses strategies for improving system security, including password management and access controls.

Why this is incorrect: It uses meta language ("This article discusses") and refers to the article instead of summarizing the content directly.

---

Bad Output 2:
The tutorial explains how to build a basic REST API using Flask. The page also highlights key configuration steps and provides a sample project structure.

Why this is incorrect: Although the first sentence is content-based, the second sentence uses meta language ("The page also highlights..."), which violates the instruction.

---

Bad Output 3:
Here is a summary of the content: A beginner-friendly introduction to the fundamentals of machine learning, covering algorithms, training data, and evaluation methods.

Why this is incorrect: It begins with meta framing ("Here is a summary of the content"), which is explicitly disallowed.

---

Good Output:
A beginner-friendly introduction to the fundamentals of machine learning, covering algorithms, training data, model evaluation, and practical considerations for getting started.
"""
