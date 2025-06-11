import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import warnings
import logging
import time
from contextlib import contextmanager
import sys
import io
import json
import fasttext
from huggingface_hub import hf_hub_download

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
                 translation_model="facebook/nllb-200-distilled-600m",
                 lid_model="facebook/fasttext-language-identification",
                 device=None,
                 suppress_warnings=True,
                 enable_translation=True):
        """
        Initialize the translation pipeline.
        
        Args:
            translation_model (str): HuggingFace model identifier for NLLB
            lid_model (str): HuggingFace model identifier for language detection
            device (str): Device to use ('cuda' or 'cpu')
            suppress_warnings (bool): Whether to suppress pipeline warnings
            enable_translation (bool): Whether to enable the translation model (True by default).
        """
        self.logger = logging.getLogger(__name__)
        self.suppress_warnings = suppress_warnings
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initializing T2TT on device: {self.device}")
        
        # Initialize Language Identification
        self.lid_model_name = lid_model
        self.logger.info(f"Loading LID model: {lid_model}")
        if self.lid_model_name == "facebook/fasttext-language-identification":
            model_path = hf_hub_download(repo_id=self.lid_model_name, filename="model.bin")
            # fasttext models run on CPU
            self.lid_model = fasttext.load_model(model_path)
            self.lid_tokenizer = None
        else:
            with suppress_stdout_stderr() if suppress_warnings else contextmanager(lambda: (yield))():
                self.lid_tokenizer = AutoTokenizer.from_pretrained(lid_model)
                self.lid_model = AutoModelForSequenceClassification.from_pretrained(lid_model).to(self.device)
            
        # Initialize Translation Pipeline only if enabled
        self.translator = None
        if enable_translation:
            translation_model = "facebook/nllb-200-distilled-600m"  # Use smaller model
            self.logger.info(f"Loading translation model: {translation_model}")
            with suppress_stdout_stderr() if suppress_warnings else contextmanager(lambda: (yield))():
                self.translator = pipeline("translation", 
                                        model=translation_model,
                                        device=self.device)
        else:
            self.logger.info("Translation model loading skipped as enable_translation is False.")
    
    def detect_language(self, text):
        """
        Detect the language of the input text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code (e.g., 'en', 'es', 'ca')
        """
        start_time = time.time()
        self.logger.info("Starting language detection")
        
        if self.lid_model_name == "facebook/fasttext-language-identification":
            # fastText expects a single line of text, clean up input
            cleaned_text = text.replace('\\n', ' ')
            predictions = self.lid_model.predict(cleaned_text, k=1)
            # The label is in format '__label__<lang_code>'
            detected_lang_code = predictions[0][0].replace('__label__', '')
            
            # We need to map back from NLLB code to 2-letter code
            nllb_to_short = {v: k for k, v in self.NLLB_LANGUAGE_CODES.items()}
            
            detected = nllb_to_short.get(detected_lang_code, detected_lang_code)
        else:
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
                
                detected = lang_map.get(detected_lang.lower(), detected_lang.lower())
            
        elapsed = time.time() - start_time
        self.logger.info(f"Language detection completed in {elapsed:.2f}s. Detected: {detected}")
        return detected
    
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
        start_time = time.time()
        self.logger.info(f"Starting translation from {source_lang} to {list(target_languages.keys())}")
        
        if not text.strip():
            self.logger.warning("Empty text provided for translation")
            return {lang: "" for lang in target_languages}
            
        translations = {}
        source_nllb = self.NLLB_LANGUAGE_CODES.get(source_lang)
        
        if not source_nllb:
            self.logger.error(f"Source language '{source_lang}' not supported for translation")
            return {lang: f"Translation failed: source language '{source_lang}' not supported" 
                   for lang in target_languages}
        
        for target_lang in target_languages:
            lang_start_time = time.time()
            
            if target_lang == source_lang:
                translations[target_lang] = text
                self.logger.info(f"Skipped translation for {target_lang} (same as source)")
                continue
                
            target_nllb = self.NLLB_LANGUAGE_CODES.get(target_lang)
            if not target_nllb:
                self.logger.error(f"Target language '{target_lang}' not supported")
                translations[target_lang] = f"Translation failed: target language '{target_lang}' not supported"
                continue
            
            try:
                with suppress_stdout_stderr() if self.suppress_warnings else contextmanager(lambda: (yield))():
                    # Split text into sentences if it's too long
                    sentences = text.split('.')
                    translated_sentences = []
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        result = self.translator(sentence.strip() + ".", 
                                              src_lang=source_nllb,
                                              tgt_lang=target_nllb,
                                              max_length=512)
                        translated_sentences.append(result[0]['translation_text'])
                    
                    translations[target_lang] = ' '.join(translated_sentences)
                    lang_elapsed = time.time() - lang_start_time
                    self.logger.info(f"Translation to {target_lang} completed in {lang_elapsed:.2f}s")
                    
            except Exception as e:
                self.logger.error(f"Translation error for {target_lang}: {str(e)}")
                translations[target_lang] = f"Translation error: {str(e)}"
        
        elapsed = time.time() - start_time
        self.logger.info(f"All translations completed in {elapsed:.2f}s")
        self.logger.debug(f"Translations:\n{json.dumps(translations, indent=2, ensure_ascii=False)}")
        
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
        start_time = time.time()
        self.logger.info("Starting text processing")
        
        detected_lang = self.detect_language(text)
        translations = self.translate_text(text, detected_lang, target_languages)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Text processing completed in {elapsed:.2f}s")
        
        return detected_lang, translations
