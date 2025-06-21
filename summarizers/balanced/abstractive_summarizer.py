from transformers import MT5ForConditionalGeneration, MT5Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from typing import List

class AbstractiveSummarizer:
    """
    Simple abstractive summarizer using dual T5 models.
    Takes extracted sentences and creates fluent, coherent summaries.
    """
    
    def __init__(self):
        """Initialize both English and multilingual T5 models."""
        print("Loading abstractive models...")
        
        # English: Fine-tuned for summarization (no prompting needed)
        self.english_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.english_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        # Multilingual: Base mT5 (requires prompting)
        self.multilingual_model = MT5ForConditionalGeneration.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
        self.multilingual_tokenizer = MT5Tokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

        print("Abstractive models loaded successfully!")
    
    def _get_models(self, language: str):
        """Get appropriate model and tokenizer based on language."""
        if language.lower() in ['en', 'english']:
            print("Using English abstractive model")
            return self.english_model, self.english_tokenizer
        else:
            print(f"Using multilingual abstractive model for {language}")
            return self.multilingual_model, self.multilingual_tokenizer
    
    def _prepare_input(self, sentences: List[str], language: str) -> str:
        text = " ".join(sentences)
        return f"summarize: {text}" 

    
    def summarize(self, sentences: List[str], language: str = 'en', max_length: int = 150) -> str:
        """
        Generate abstractive summary from extracted sentences.
        
        Args:
            sentences: List of key sentences from extractive summarizer
            language: Language code ('en' for English, others for multilingual)
            max_length: Maximum length of generated summary
            
        Returns:
            Abstractive summary as string
        """
        if not sentences:
            return ""
        
        # Get appropriate model and tokenizer
        model, tokenizer = self._get_models(language)
        
        # Prepare input text
        input_text = self._prepare_input(sentences, language)
        
        # Tokenize input (512 is input limit, not output)
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        outputs = model.generate(
            inputs,
            max_length=max_length,
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode output
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary.strip()