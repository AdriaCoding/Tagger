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

class TextEmbeddingTagger(BaseTagger):
    """
    Tagger que utiliza embeddings de texto a través de SentenceTransformer
    Requiere transcripción previa del audio
    """
    
    def __init__(self, taxonomy_file, model_name='paraphrase-multilingual-mpnet-base-v2',
                 S2TT_model="openai/whisper-small", device=None,
                 decision_method=DECISION_METHOD_KNN, decision_params=None):
        """
        Inicializa el tagger basado en embeddings de texto.
        
        Args:
            taxonomy_file (str): Ruta al archivo de taxonomía
            model_name (str): Nombre del modelo de embeddings de texto
            S2TT_model (str): Modelo para transcribir audio al inglés
            device (str): Dispositivo a utilizar
            decision_method (str): Método para seleccionar etiquetas
            decision_params (dict): Parámetros adicionales para el método de selección
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.S2TT_model = S2TT_model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializar modelo de embeddings
        start_time = time.time()
        self.logger.info(f"Loading embeddings model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        elapsed = time.time() - start_time
        self.logger.info(f"Embeddings model loaded in {elapsed:.6f}s")
        
        # No inicializamos ASR ni translator aquí
        self.asr = None
        self.translator = None
        
        # Inicializar clase base
        super().__init__(taxonomy_file, self.device, decision_method, decision_params)
    
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

        # Get language from kwargs, defaulting to None (for auto-detection)
        language = kwargs.get('language')

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

            # Get embedding for the sample using the original transcription
            self.logger.info("Computing embedding using original transcription.")
            sample_embedding, _ = self.get_audio_embedding(audio_path=sample_path, transcription=original_transcription)
            
            # Find similar tags
            self.logger.info("Finding similar tags")
            nearest_tags, similarities = self.find_similar_tags(sample_embedding)
            
            # Create result
            result = {
                'file': os.path.basename(sample_path),
                'transcription': original_transcription, # Original language
                'transcription_eng': "", # English transcription
                'lang': detected_lang_code, # Detected language code
                'tags': []
            }
            
            # Add tags with similarities
            for i in range(len(nearest_tags)):
                result['tags'].append({
                    'tag': nearest_tags[i],
                    'similarity': similarities[i]
                })
            
            elapsed = time.time() - start_time
            self.logger.info(f"Sample tagging completed in {elapsed:.6f}s")
            self.logger.debug(f"Tagging result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return result
        finally:
            self._unload_asr() # Ensure ASR model is unloaded after all transcriptions are done 