import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
import time
import json
from .S2TT import WhisperS2TT
from .base_tagger import BaseTagger, DECISION_METHOD_KNN
from .T2TT import T2TT
from .tagging_strategy import TaggingStrategy, SemanticTaggingStrategy, create_tagging_strategy


class TextEmbeddingTagger(BaseTagger):
    """
    Tagger que utiliza embeddings de texto a través de SentenceTransformer
    Requiere transcripción previa del audio.
    
    Supports two tagging strategies:
    - Semantic: Uses SentenceTransformer embeddings + KNN/radius matching
    - Syntactic: Uses YAKE keyword extraction
    """
    
    def __init__(self, taxonomy_file, model_name='paraphrase-multilingual-mpnet-base-v2',
                 S2TT_model="openai/whisper-small", device=None,
                 decision_method=DECISION_METHOD_KNN, decision_params=None,
                 tagging_strategy=None, tagging_strategy_type='semantic', **strategy_kwargs):
        """
        Inicializa el tagger basado en embeddings de texto.
        
        Args:
            taxonomy_file (str): Ruta al archivo de taxonomía
            model_name (str): Nombre del modelo de embeddings de texto
            S2TT_model (str): Modelo para transcribir audio al inglés
            device (str): Dispositivo a utilizar
            decision_method (str): Método para seleccionar etiquetas
            decision_params (dict): Parámetros adicionales para el método de selección
            tagging_strategy (TaggingStrategy): Pre-created tagging strategy instance
            tagging_strategy_type (str): Type of strategy to create if tagging_strategy is None
                                         ('semantic' or 'syntactic')
            **strategy_kwargs: Additional arguments for strategy creation
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.S2TT_model = S2TT_model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tagging_strategy_type = tagging_strategy_type

        # Store strategy kwargs for later use
        self._strategy_kwargs = strategy_kwargs
        
        # Handle tagging strategy
        if tagging_strategy is not None:
            # Use provided strategy
            self.tagging_strategy = tagging_strategy
            self.logger.info(f"Using provided tagging strategy: {type(tagging_strategy).__name__}")
            
            # For semantic strategy, we still need the embedding model for BaseTagger
            if isinstance(tagging_strategy, SemanticTaggingStrategy):
                self.embedding_model = tagging_strategy.embedding_model
            else:
                # For syntactic strategy, we need to load embedding model for BaseTagger compatibility
                start_time = time.time()
                self.logger.info(f"Loading embeddings model for BaseTagger: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                elapsed = time.time() - start_time
                self.logger.info(f"Embeddings model loaded in {elapsed:.6f}s")
        else:
            # Create strategy based on type
            if tagging_strategy_type == 'syntactic':
                # For syntactic, create strategy without taxonomy (it doesn't need it)
                self.tagging_strategy = create_tagging_strategy(
                    'syntactic',
                    taxonomy_file=None,
                    **strategy_kwargs
                )
                # Still need embedding model for BaseTagger compatibility
                start_time = time.time()
                self.logger.info(f"Loading embeddings model for BaseTagger: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                elapsed = time.time() - start_time
                self.logger.info(f"Embeddings model loaded in {elapsed:.6f}s")
            else:
                # For semantic, create strategy with taxonomy
                # First load the embedding model (shared with strategy)
                start_time = time.time()
                self.logger.info(f"Loading embeddings model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                elapsed = time.time() - start_time
                self.logger.info(f"Embeddings model loaded in {elapsed:.6f}s")
                
                # Strategy will be created after BaseTagger init (it needs taxonomy path)
                self.tagging_strategy = None
        
        # No inicializamos ASR ni translator aquí
        self.asr = None
        self.translator = None
        
        # Inicializar clase base (this sets up taxonomy_file path, tags, embeddings, etc.)
        super().__init__(taxonomy_file, self.device, decision_method, decision_params)
        
        # Now create semantic strategy if needed (after BaseTagger has set up paths)
        if self.tagging_strategy is None and tagging_strategy_type == 'semantic':
            self.tagging_strategy = create_tagging_strategy(
                'semantic',
                taxonomy_file=taxonomy_file,
                model_name=model_name,
                device=self.device,
                decision_method=decision_method,
                decision_params=decision_params
            )
            # Share the embedding model to avoid loading twice
            self.tagging_strategy.embedding_model = self.embedding_model
    
    def _load_asr(self):
        """Carga el modelo ASR bajo demanda"""
        if self.asr is None:
            start_time = time.time()
            self.logger.info(f"Loading ASR model: {self.S2TT_model}")
            self.asr = WhisperS2TT(model_name=self.S2TT_model, device=self.device)
            elapsed = time.time() - start_time
            self.logger.info(f"ASR model loaded in {elapsed:.6f}s")
    
    def _unload_asr(self):
        """Libera la memoria del modelo ASR"""
        if self.asr is not None:
            del self.asr
            self.asr = None
            torch.cuda.empty_cache()
            self.logger.info("ASR model unloaded and memory cleared")
    
    def _load_translator(self, enable_translation=True):
        """Carga el modelo de traducción bajo demanda"""
        if self.translator is None:
            start_time = time.time()
            self.logger.info("Initializing translation pipeline")
            self.translator = T2TT(device=self.device, enable_translation=enable_translation)
            elapsed = time.time() - start_time
            self.logger.info(f"Translation pipeline initialized in {elapsed:.6f}s")
    
    def get_model_identifier(self):
        """
        Devuelve identificador único para este modelo.
        
        Returns:
            str: Identificador del modelo
        """
        return f"text_{self.model_name}"
    
    def transcribe_audio(self, audio_file, language=None):
        """
        Transcribe un archivo de audio.
        
        Args:
            audio_file (str): Ruta al archivo de audio
            language (str, optional): Código de idioma
            
        Returns:
            str: Texto transcrito
        """
        # The ASR model is loaded and unloaded in tag_sample, so we just use it here.
        start_time = time.time()
        self.logger.info(f"Starting audio transcription: {audio_file}")
        result = self.asr.transcribe(audio_file, language=language)
        elapsed = time.time() - start_time
        self.logger.info(f"Audio transcription completed in {elapsed:.6f}s")
        return result["text"]
    
    def get_tag_embedding(self, tag):
        """
        Obtiene el embedding para una etiqueta.
        
        Args:
            tag (str): Texto de la etiqueta
            
        Returns:
            numpy.ndarray: Vector de embedding
        """
        start_time = time.time()
        self.logger.debug(f"Computing embedding for tag: {tag}")
        
        embedding = self.embedding_model.encode(tag)
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Tag embedding computed in {elapsed:.6f}s")
        return embedding
    
    def get_audio_embedding(self, audio_path, transcription=None, language=None, translations=None):
        """
        Obtiene el embedding para una muestra de audio.
        
        Args:
            audio_path (str): Ruta al archivo de audio
            transcription (str, optional): Transcripción existente
            language (str, optional): Código de idioma
            
        Returns:
            numpy.ndarray: Vector de embedding
            string: Transcripción
        """
        start_time = time.time()
        self.logger.info(f"Computing audio embedding for: {audio_path}")
        
        # Si no hay transcripción, transcribir audio
        if transcription is None:
            self.logger.info("No transcription provided, transcribing audio")
            transcription = self.transcribe_audio(audio_path, language)
        
        # Calcular embedding para la transcripción
        embedding = self.embedding_model.encode(transcription)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Audio embedding computed in {elapsed:.6f}s")
        return embedding, transcription
    
    def tag_text(self, text: str, top_k: int = 5) -> list:
        """
        Tag text directly using the configured tagging strategy.
        
        This method allows tagging text without going through audio transcription.
        
        Args:
            text (str): The text to tag
            top_k (int): Maximum number of tags to return
            
        Returns:
            list: List of dicts with 'tag' and 'similarity' keys
        """
        return self.tagging_strategy.tag_text(text, top_k=top_k)
    
    def tag_sample(self, sample_path, translation_languages=None, **kwargs):
        """
        Etiqueta una muestra.
        
        Args:
            sample_path (str): Ruta a la muestra
            translation_languages (dict): Dictionary of target languages {code: name}
            **kwargs: Argumentos adicionales específicos del modelo
            
        Returns:
            dict: Diccionario con resultados
        """
        start_time = time.time()
        self.logger.info(f"Starting sample tagging: {sample_path}")
        self.logger.info(f"Using tagging strategy: {type(self.tagging_strategy).__name__}")

        # Get language from kwargs, defaulting to None (for auto-detection)
        language = kwargs.get('language')
        top_k = kwargs.get('top_k', 5)

        # Load ASR model once for both transcriptions
        self._load_asr()
        
        try:
            # Get original language transcription
            self.logger.info(f"Transcribing audio to original language (language: {language or 'auto-detect'}): {sample_path}")
            original_transcription_result = self.asr.transcribe(sample_path, language=language)
            original_transcription = original_transcription_result['text']
            self.logger.info(f"Original Transcription: {original_transcription}")

            # Load and run LID
            self._load_translator(enable_translation=False) # Ensure translator is not loaded
            detected_lang_code = self.translator.detect_language(original_transcription)
            self.logger.info(f"Detected language: {detected_lang_code}")
            
            # Update syntactic strategy language if using syntactic tagging
            if self.tagging_strategy_type == 'syntactic' and detected_lang_code:
                # Map detected language code to YAKE language code if needed
                self.tagging_strategy.language = detected_lang_code
                self.logger.info(f"Updated syntactic strategy language to: {detected_lang_code}")

            # Use tagging strategy to get tags from transcription
            self.logger.info("Tagging transcription using strategy")
            tag_results = self.tagging_strategy.tag_text(original_transcription, top_k=top_k)
            
            # Create result
            result = {
                'file': os.path.basename(sample_path),
                'transcription': original_transcription, # Original language
                'transcription_eng': "", # English transcription
                'lang': detected_lang_code, # Detected language code
                'tagging_strategy': self.tagging_strategy_type,
                'tags': tag_results
            }
            
            elapsed = time.time() - start_time
            self.logger.info(f"Sample tagging completed in {elapsed:.6f}s")
            self.logger.debug(f"Tagging result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return result
        finally:
            self._unload_asr() # Ensure ASR model is unloaded after all transcriptions are done
