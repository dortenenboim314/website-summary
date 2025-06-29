from summarizers import AbstractSummarizer, MarkdownPreprocessor
from summarizers.light import SpacyTextrank
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

class MT5(AbstractSummarizer):
    def __init__(self):
        print("MT5 v2")
        self.model = MT5ForConditionalGeneration.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
        self.tokenizer = MT5Tokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
        self.candidate_selector = SpacyTextrank()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
            
    def summarize(self, text: str, lang: str, **kwargs) -> str:
        should_filter = kwargs.get("should_filter", True)
        if should_filter:
            # Filter text using candidate selector
            print("Filtering text using candidate selector...")
            text = self.candidate_selector.summarize(text, lang=lang, limit_sentences=5)
        else:
            print("Markdown preprocessing without filtering...")
            text = MarkdownPreprocessor.markdown_to_clean_text(text)
            
        input_ids = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=200,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]

        summary = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return summary