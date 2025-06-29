from summarizers import AbstractSummarizer
import anthropic


class ClaudeSummarizer(AbstractSummarizer):
    def __init__(self, model_name = "claude-3-5-haiku-20241022"):
        print("claude summarizer v3")
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        # print(self.system_message.format(lang=lang))
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=80,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _extractive_prompt.format(text=text)
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
_extractive_prompt = """
The following is the full plain text of a webpage. It may contain irrelevant sections such as headers, navigation links, author bios, newsletter banners, or footers.

{text}

---

Now, based only on the meaningful content above, select the 2-3 most important sentences that summarize the main ideas. 

- Do NOT rephrase or paraphrase—output each sentence exactly as it appears in the text.
- Return them in the order they appear.
- Do NOT add any extra text, numbering, or commentary.
"""
