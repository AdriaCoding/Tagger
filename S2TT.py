import torch
from transformers import pipeline
import argparse
import os
import warnings
import textwrap
import sys
import io
import logging
import time
import json
from contextlib import contextmanager
from transformers import logging as transformers_logging

# Lista de modelos Whisper disponibles
SUPPORTED_MODELS = [
    "openai/whisper-tiny",      # 39M parámetros
    "openai/whisper-base",      # 74M parámetros
    "openai/whisper-small",     # 244M parámetros
    #"openai/whisper-medium",    # 769M parámetros
    "openai/whisper-large-v3-turbo",  # versión optimizada para mejor velocidad
    "openai/whisper-large-v3"  # 1550M parámetros, mejor rendimiento
]

# Contextmanager para suprimir stdout/stderr temporalmente
@contextmanager
def suppress_stdout_stderr():
    """
    Contexto que suprime temporalmente salidas a stdout y stderr.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stderr
        sys.stderr = old_stderr

class WhisperS2TT:
    """
    Clase para realizar transcripción de audio al inglés utilizando el modelo Whisper.
    """
    
    def __init__(self, model_name="openai/whisper-large-v3", device=None, suppress_warnings=True):
        """
        Inicializa el modelo Whisper para transcripción de audio.
        """
        self.logger = logging.getLogger(__name__)
        
        # Suprimir warnings si se solicita
        self.suppress_warnings = suppress_warnings
        if suppress_warnings:
            warnings.filterwarnings("ignore")
            transformers_logging.set_verbosity_error()
            old_environ = os.environ.copy()
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Autodetectar dispositivo si no se especifica
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"Device set to use {self.device}")
        
        # Inicializar pipeline de ASR, suprimiendo salidas si es necesario
        start_time = time.time()
        self.logger.info(f"Loading ASR model: {model_name}")
        
        if suppress_warnings:
            with suppress_stdout_stderr():
                self.asr_model = pipeline(
                    task="automatic-speech-recognition",
                    model=model_name,
                    device=self.device
                )
        else:
            self.asr_model = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device=self.device
            )
        
        elapsed = time.time() - start_time
        self.logger.info(f"ASR model loaded in {elapsed:.2f}s")
        self.model_name = model_name
        
    def transcribe(self, audio_file, language=None, **kwargs):
        """
        Transcribe un archivo de audio a texto.
        """
        start_time = time.time()
        self.logger.info(f"Starting transcription of {audio_file}")
        
        generate_kwargs = kwargs.pop('generate_kwargs', {})
        
        # Si se especifica un idioma, usarlo
        if language:
            generate_kwargs['language'] = language
            self.logger.info(f"Using specified language: {language}")
            
        # Asegurarse de que return_timestamps esté habilitado para archivos largos
        generate_kwargs['return_timestamps'] = True
        
        try:
            # Transcribir audio, suprimiendo salidas si es necesario
            if self.suppress_warnings:
                with suppress_stdout_stderr():
                    result = self.asr_model(audio_file, generate_kwargs=generate_kwargs, **kwargs)
            else:
                result = self.asr_model(audio_file, generate_kwargs=generate_kwargs, **kwargs)
            
            # Si el resultado incluye timestamps, extraer solo el texto
            if isinstance(result, dict) and 'text' in result:
                final_result = result
            elif isinstance(result, str):
                final_result = {"text": result}
            else:
                final_result = {"text": str(result)}
            
            elapsed = time.time() - start_time
            self.logger.info(f"Transcription completed in {elapsed:.2f}s")
            self.logger.debug(f"Transcription result:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}")
            
            return final_result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Transcription failed after {elapsed:.2f}s: {str(e)}")
            raise
    
    def get_info(self):
        """
        Obtiene información sobre el modelo ASR actual.
        """
        info = {
            "model_name": self.model_name,
            "device": self.device
        }
        self.logger.debug(f"Model info: {json.dumps(info, indent=2)}")
        return info

# Función simple para transcribir un archivo de audio sin crear una instancia de clase
def transcribe_audio(audio_file, model_name="openai/whisper-large-v3", language=None, device=None, suppress_warnings=True):
    """
    Transcribe un archivo de audio usando Whisper sin crear una instancia de clase.
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Crear instancia temporal de WhisperASR
        logger.info(f"Creating temporary ASR instance with model {model_name}")
        asr = WhisperS2TT(model_name=model_name, device=device, suppress_warnings=suppress_warnings)
        
        # Transcribir audio
        result = asr.transcribe(audio_file, language=language)
        
        elapsed = time.time() - start_time
        logger.info(f"Standalone transcription completed in {elapsed:.2f}s")
        logger.debug(f"Transcription result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result["text"]
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Standalone transcription failed after {elapsed:.2f}s: {str(e)}")
        raise

# Uso de ejemplo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(elapsed_time).3fs] - %(message)s'
    )
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Configurar el parser de argumentos con RawTextHelpFormatter para preservar saltos de línea
    parser = argparse.ArgumentParser(
        description="Transcribe audio usando el modelo Whisper",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumentos disponibles
    parser.add_argument("--audio_file_path", type=str, help="Ruta relativa al archivo de audio", 
                       default="audios_test\\barcelona_alexdobano_m68284_a85028_audio_converted.wav")
    parser.add_argument("--language", type=str, help="Idioma de transcripción (ej: en, es, fr). Dejar vacío para auto-detección", 
                       default=None)
    
    # Crear el mensaje de ayuda para los modelos
    model_help = "Modelo Whisper a utilizar. Opciones disponibles:\n"
    for model in SUPPORTED_MODELS:
        model_help += f"  - {model}\n"
    
    parser.add_argument("--model_name", type=str, 
                       help=model_help,
                       default="openai/whisper-small")
    parser.add_argument("--device", type=str, 
                       help="Dispositivo a utilizar (cuda, cpu). Si no se especifica, se autodetecta", 
                       default=None)
    parser.add_argument("--silent", action="store_true", 
                       help="No mostrar advertencias durante la ejecución")
    
    # Parsear argumentos
    args = parser.parse_args()
    logger.info(f"Arguments parsed: {vars(args)}")
    
    # Convertir ruta relativa a absoluta si es necesario
    audio_path = args.audio_file_path
    if not os.path.isabs(audio_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        audio_path = os.path.join(base_dir, audio_path)
    
    # Verificar si el archivo existe
    if os.path.exists(audio_path):
        logger.info(f"Processing file: {audio_path}")
        logger.info(f"Using model: {args.model_name}")
        
        if args.language:
            logger.info(f"Language specified: {args.language}")
        else:
            logger.info("Mode: Automatic language detection")
        
        try:
            # Transcribir audio
            transcription = transcribe_audio(audio_path, model_name=args.model_name, 
                                          language=args.language, device=args.device,
                                          suppress_warnings=args.silent)
            
            # Mostrar resultados
            logger.info(f"\nTranscription: {transcription}")
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            sys.exit(1)
            
    else:
        logger.error(f"Error: File {audio_path} does not exist.")
        logger.info("Verify the path or provide an absolute path with --audio_file_path.")
        sys.exit(1)
        
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
