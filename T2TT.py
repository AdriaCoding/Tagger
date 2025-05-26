import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import warnings
import logging
from contextlib import contextmanager
import sys
import io

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout/stderr"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

class T2TT:
    """
    Text to Translated Text: Handles language identification and translation
    """
    
    # NLLB language codes mapping
    NLLB_LANGUAGE_CODES = {
        "en": "eng_Latn",
        "es": "spa_Latn",
        "ca": "cat_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "pt": "por_Latn",
        "it": "ita_Latn"
    }

    def __init__(self, 
                 translation_model="facebook/nllb-200-distilled-600M",
                 lid_model="papluca/xlm-roberta-base-language-detection",
                 device=None,
                 suppress_warnings=True):
        """
        Initialize the translation pipeline.
        
        Args:
            translation_model (str): HuggingFace model identifier for NLLB
            lid_model (str): HuggingFace model identifier for language detection
            device (str): Device to use ('cuda' or 'cpu')
            suppress_warnings (bool): Whether to suppress pipeline warnings
        """
        self.suppress_warnings = suppress_warnings
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing T2TT on device: {self.device}")
        
        # Initialize Language Identification
        print(f"Loading LID model: {lid_model}")
        with suppress_stdout_stderr() if suppress_warnings else contextmanager(lambda: (yield))():
            self.lid_tokenizer = AutoTokenizer.from_pretrained(lid_model)
            self.lid_model = AutoModelForSequenceClassification.from_pretrained(lid_model).to(self.device)
            
        # Initialize Translation Pipeline
        print(f"Loading translation model: {translation_model}")
        with suppress_stdout_stderr() if suppress_warnings else contextmanager(lambda: (yield))():
            self.translator = pipeline("translation", 
                                    model=translation_model,
                                    device=self.device)
    
    def detect_language(self, text):
        """
        Detect the language of the input text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code (e.g., 'en', 'es', 'ca')
        """
        with suppress_stdout_stderr() if self.suppress_warnings else contextmanager(lambda: (yield))():
            inputs = self.lid_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.lid_model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)
                
            detected_lang_id = predictions.argmax().item()
            detected_lang = self.lid_model.config.id2label[detected_lang_id]
            
            # Map full language names or codes to our standard short codes
            lang_map = {
                "en": "en", "english": "en",
                "es": "es", "spanish": "es", "español": "es",
                "ca": "ca", "catalan": "ca", "català": "ca",
                # Add more mappings as needed
            }
            
            return lang_map.get(detected_lang.lower(), detected_lang.lower())
    
    def translate_text(self, text, source_lang, target_languages):
        """
        Translate text to multiple target languages.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code (e.g., 'en')
            target_languages (dict): Dictionary of target language codes and their names
            
        Returns:
            dict: Dictionary of translations {lang_code: translated_text}
        """
        if not text.strip():
            return {lang: "" for lang in target_languages}
            
        translations = {}
        source_nllb = self.NLLB_LANGUAGE_CODES.get(source_lang)
        
        if not source_nllb:
            print(f"Warning: Source language '{source_lang}' not supported for translation")
            return {lang: f"Translation failed: source language '{source_lang}' not supported" 
                   for lang in target_languages}
        
        for target_lang in target_languages:
            if target_lang == source_lang:
                translations[target_lang] = text
                continue
                
            target_nllb = self.NLLB_LANGUAGE_CODES.get(target_lang)
            if not target_nllb:
                translations[target_lang] = f"Translation failed: target language '{target_lang}' not supported"
                continue
            
            try:
                with suppress_stdout_stderr() if self.suppress_warnings else contextmanager(lambda: (yield))():
                    result = self.translator(text, 
                                          src_lang=source_nllb,
                                          tgt_lang=target_nllb,
                                          max_length=512)
                    translations[target_lang] = result[0]['translation_text']
            except Exception as e:
                translations[target_lang] = f"Translation error: {str(e)}"
        
        return translations
    
    def process_text(self, text, target_languages):
        """
        Main processing function: detects language and translates to all target languages.
        
        Args:
            text (str): Input text
            target_languages (dict): Dictionary of target language codes and their names
            
        Returns:
            tuple: (detected_language, translations_dict)
        """
        detected_lang = self.detect_language(text)
        translations = self.translate_text(text, detected_lang, target_languages.keys())
        return detected_lang, translations
