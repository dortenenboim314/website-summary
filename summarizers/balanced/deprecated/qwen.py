from summarizers import AbstractSummarizer, MarkdownPreprocessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen(AbstractSummarizer):
    def __init__(self):
        print("Qwen v1.1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
        ).eval().to('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        
            
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        text = MarkdownPreprocessor.markdown_to_clean_text(text)
        messages = [
        {
            "role": "system",
            "content": (
            "You are an expert summarization assistant. "
            "When given plain text extracted from a webpage, you will produce "
            "a concise, factual summary *only*—no preamble or commentary. "
            "Your summary must be in the same language as the input text, "
            )
        },
        {
            "role": "user",
            "content": (
            "Please summarize the following text. "
            "Do not add any extra words or introductions:\n\n"
            f"{text}"
            )
        }
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]


        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response