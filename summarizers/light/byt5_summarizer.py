from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from summarizers import MarkdownPreprocessor
from summarizers.light import SumyTextRankSummarizer

class ByT5ONNXSummarizer:
    def __init__(
        self,
        model_name="google/byt5-small",
        max_input_tokens=1024,
        max_output_tokens=256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)  # Exports to ONNX if needed

        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.reduce_sentences = max_input_tokens // 32  # rough ~tokens per sentence

        self.reducer = SumyTextRankSummarizer()

    def summarize(self, markdown: str, num_sentences: int, language: str = "en") -> str:
        text = MarkdownPreprocessor.markdown_to_clean_text(markdown)

        # Reduce input if it's too long
        token_count = len(self.tokenizer.tokenize(text))
        if token_count > self.max_input_tokens:
            text = self.reducer.summarize(markdown, language, self.reduce_sentences)

        prompt = f"summarize: {text}"
        
        # prompt = f"Summarize the following in {num_sentences} concise sentences:\n\n{text}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=2 ** 16,  # Set to a large number to avoid truncation
            num_beams=1,
            # early_stopping=True,
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
