from summarizers import AbstractSummarizer, MarkdownPreprocessor
from summarizers.light import SpacyTextrank
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
from torch.quantization import quantize_dynamic
import torch

class BartLargeSummarizer(AbstractSummarizer):
    def __init__(self):
        print("bart-large-cnn v8")
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.candidate_selector = SpacyTextrank()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
            
    def summarize(self, text: str, lang: str = "en", **kwargs) -> str:
        if lang != "english":
            print(f"Language '{lang}' is not supported.")
            return ""
        
        should_filter = kwargs.get("should_filter", True)
        if should_filter:
            # Filter text using candidate selector
            print("Filtering text using candidate selector...")
            text = self.candidate_selector.summarize(text, lang=lang, limit_sentences=10)
        else:
            print("Markdown preprocessing without filtering...")
            text = MarkdownPreprocessor.markdown_to_clean_text(text)
        
        # Tokenize input with truncation to 1024 tokens
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200,
                min_length=30,
                do_sample=False,
                num_beams=1,
                early_stopping=True
            )
        
        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary