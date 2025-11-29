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
import subprocess
import tempfile

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
        self.logger.info(f"ASR model loaded in {elapsed:.6f}s")
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
        
        # Convert audio to WAV format to handle container/extension mismatches
        converted_file = None
        file_to_process = audio_file
        
        try:
            # Try to convert to WAV using ffmpeg for reliable format handling
            converted_file = self._convert_to_wav(audio_file)
            if converted_file:
                file_to_process = converted_file
                self.logger.info(f"Using converted WAV file: {converted_file}")
            
            # Transcribir audio, suprimiendo salidas si es necesario
            if self.suppress_warnings:
                with suppress_stdout_stderr():
                    result = self.asr_model(file_to_process, return_timestamps=True, generate_kwargs=generate_kwargs, **kwargs)
            else:
                result = self.asr_model(file_to_process, return_timestamps=True, generate_kwargs=generate_kwargs, **kwargs)
            
            # Si el resultado incluye timestamps, extraer solo el texto
            if isinstance(result, dict) and 'text' in result:
                final_result = result
            elif isinstance(result, str):
                final_result = {"text": result}
            else:
                final_result = {"text": str(result)}
            
            elapsed = time.time() - start_time
            self.logger.info(f"Transcription completed in {elapsed:.6f}s")
            self.logger.debug(f"Transcription result:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}")
            
            return final_result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Transcription failed after {elapsed:.6f}s: {str(e)}")
            raise
        finally:
            # Clean up temporary converted file
            if converted_file and os.path.exists(converted_file):
                try:
                    os.remove(converted_file)
                    self.logger.debug(f"Cleaned up temporary file: {converted_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {converted_file}: {e}")
    
    def _convert_to_wav(self, audio_file):
        """
        Convert audio file to WAV format using ffmpeg.
        This handles container/extension mismatches (e.g., M4A files with .mp3 extension).
        
        Args:
            audio_file (str): Path to the input audio file
            
        Returns:
            str: Path to the converted WAV file, or None if conversion fails
        """
        try:
            # Create a temporary file for the converted audio
            fd, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Use ffmpeg to convert to WAV (16kHz mono, which is what Whisper expects)
            cmd = [
                'ffmpeg',
                '-i', audio_file,
                '-ar', '16000',      # Sample rate 16kHz
                '-ac', '1',          # Mono
                '-y',                # Overwrite output
                '-loglevel', 'error',
                temp_wav
            ]
            
            self.logger.info(f"Converting audio to WAV format: {audio_file}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"ffmpeg conversion failed: {result.stderr}")
                # Clean up failed temp file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                return None
            
            self.logger.info(f"Audio converted successfully to: {temp_wav}")
            return temp_wav
            
        except FileNotFoundError:
            self.logger.warning("ffmpeg not found, skipping audio conversion")
            return None
        except subprocess.TimeoutExpired:
            self.logger.warning("ffmpeg conversion timed out")
            return None
        except Exception as e:
            self.logger.warning(f"Audio conversion failed: {e}")
            return None
    
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
        logger.info(f"Standalone transcription completed in {elapsed:.6f}s")
        logger.debug(f"Transcription result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result["text"]
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Standalone transcription failed after {elapsed:.6f}s: {str(e)}")
        raise

# Uso de ejemplo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
                       default="../audios_test/barcelona_alexdobano_m68284_a85028_audio_converted.wav")
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
        audio_path = os.path.abspath(os.path.join(os.getcwd(), audio_path))
    
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
    logger.info(f"Total execution time: {total_time:.6f} seconds")
