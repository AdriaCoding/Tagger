import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from .S2TT import WhisperS2TT
from .base_tagger import BaseTagger, DECISION_METHOD_KNN

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
        self.model_name = model_name
        self.S2TT_model = S2TT_model
        
        # Inicializar modelo de embeddings
        print(f"Cargando modelo de embeddings: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Inicializar ASR para transcripción usando WhisperS2TT
        print(f"Cargando modelo ASR: {S2TT_model}...")
        if device is None:
            device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_to_use = device
        
        self.asr = WhisperS2TT(model_name=S2TT_model, device=device_to_use)
        
        # Inicializar clase base
        super().__init__(taxonomy_file, device, decision_method, decision_params)
    
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
        # Usar la interfaz de WhisperS2TT para transcribir
        result = self.asr.transcribe(audio_file, language=language)
        return result["text"]
    
    def get_tag_embedding(self, tag):
        """
        Obtiene el embedding para una etiqueta.
        
        Args:
            tag (str): Texto de la etiqueta
            
        Returns:
            numpy.ndarray: Vector de embedding
        """
        return self.embedding_model.encode(tag)
    
    def get_audio_embedding(self, audio_path, transcription=None, language=None):
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
        # Si no hay transcripción, transcribir audio
        if transcription is None:
            print(f"Transcribiendo audio: {audio_path}")
            transcription = self.transcribe_audio(audio_path, language)
        
        # Calcular embedding para la transcripción
        return self.embedding_model.encode(transcription), transcription
    
    def tag_sample(self, sample_path, **kwargs):
        """
        Etiqueta una muestra.
        
        Args:
            sample_path (str): Ruta a la muestra
            **kwargs: Argumentos adicionales específicos del modelo
            
        Returns:
            dict: Diccionario con resultados
        """
        # Obtener embedding para la muestra
        sample_embedding, transcription = self.get_audio_embedding(sample_path, **kwargs)
        
        # Encontrar etiquetas similares según el método configurado
        nearest_tags, similarities = self.find_similar_tags(sample_embedding)
        
        # Crear resultado
        result = {
            'file': os.path.basename(sample_path),
            'transcription': transcription,
            'tags': []
        }
        
        # Añadir tags con similitudes
        for i in range(len(nearest_tags)):
            result['tags'].append({
                'tag': nearest_tags[i],
                'similarity': similarities[i]
            })
        
        return result 