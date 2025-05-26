import argparse
import os
import json
import logging
import time
from datetime import datetime
from .base_tagger import (
    DECISION_METHOD_KNN,
    DECISION_METHOD_RADIUS,
    DECISION_METHOD_ADAPTIVE
)
from .factory import create_tagger

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'tagger_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - [%(elapsed_time).3fs] - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Add elapsed time to log record
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.elapsed_time = time.time() - start_time
        return record
    logging.setLogRecordFactory(record_factory)

def main():
    # Start timing
    global start_time
    start_time = time.time()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Tagger application")

    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Etiquetar un archivo de audio con diferentes modelos")
    
    parser.add_argument("--audio_file", type=str, required=True,
                      help="Ruta al archivo de audio a etiquetar")
    parser.add_argument("--tagger_type", type=str, choices=['text', 'audio', 'hybrid'], default='text',
                      help="Tipo de tagger a utilizar: text (SentenceTransformer), audio (CLAP), hybrid (combinación)")
    parser.add_argument("--taxonomy_file", type=str, default="16tags.txt",
                      help="Ruta al archivo de taxonomía de etiquetas")
    parser.add_argument("--text_model", type=str, default="paraphrase-multilingual-mpnet-base-v2",
                      help="Modelo de embeddings de texto")
    parser.add_argument("--audio_model", type=str, default="facebook/hubert-base-ls960",
                      help="Modelo de embeddings de audio")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused",
                      help="Modelo CLAP para embeddings de audio")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small",
                      help="Modelo Whisper para ASR")
    parser.add_argument("--audio_weight", type=float, default=0.3,
                      help="Peso para embeddings de audio en modo híbrido")
    parser.add_argument("--text_weight", type=float, default=0.7,
                      help="Peso para embeddings de texto en modo híbrido")
    parser.add_argument("--language", type=str, default=None,
                      help="Idioma para transcripción (ej: es, en)")
    parser.add_argument("--json_output", action="store_true",
                      help="Mostrar resultado en formato JSON en la consola")
    parser.add_argument('--decision_method', type=str, default=DECISION_METHOD_KNN,
                      choices=[DECISION_METHOD_KNN, DECISION_METHOD_RADIUS, DECISION_METHOD_ADAPTIVE],
                      help='Método de decisión para seleccionar etiquetas')
    parser.add_argument('--decision_params', type=json.loads, default='{}',
                      help='Parámetros para el método de decisión en formato JSON')
    parser.add_argument("--top_k", type=int, default=None,
                      help="Número de etiquetas a devolver (para KNN)")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Threshhold similaridad para el método Radius Nearest Neighbors")
    parser.add_argument("--min_threshold", type=float, default=None,
                      help="Threshold mínimo para método adaptativo (devuelve tags > threshold o el mejor)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                      help="Dispositivo a utilizar para los cálculos (cpu o cuda)")
    
    args = parser.parse_args()

    # Configurar parámetros según el tipo de tagger
    tagger_params = {}
    decision_params = {}
    
    decision_params = {
        'k': args.top_k,
        'threshold': args.threshold, 
        'min_threshold': args.min_threshold
    }
    # Asignar solo los parámetros que no son None
    decision_params = {k:v for k,v in decision_params.items() if v is not None}

    if args.tagger_type == 'text':
        tagger_params = {
            'model_name': args.text_model,
            'S2TT_model': args.whisper_model,
            'decision_method': args.decision_method,
            'decision_params': decision_params,
            'device': args.device
        }
    elif args.tagger_type == 'audio':
        tagger_params = {
            'model_name': args.clap_model,
            'decision_method': args.decision_method,
            'decision_params': decision_params,
            'device': args.device
        }
    elif args.tagger_type == 'hybrid':
        tagger_params = {
            'audio_model_name': args.audio_model,
            'text_model_name': args.text_model,
            'asr_model_name': args.whisper_model,
            'audio_weight': args.audio_weight,
            'text_weight': args.text_weight,
            'decision_method': args.decision_method,
            'decision_params': decision_params,
            'device': args.device
        }
    
    # Crear tagger
    tagger = create_tagger(args.tagger_type, args.taxonomy_file, **tagger_params)
    
    # Procesar archivo de audio individual
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        raise FileNotFoundError(f"El archivo de audio {args.audio_file} no existe")
                
    kwargs = {'language': args.language}

    translation_languages = {
        'en': 'English',
        'es': 'Español',
        'ca': 'Català'
    }

    logger.info(f"Processing audio file: {args.audio_file}")
    result = tagger.tag_sample(args.audio_file, translation_languages, **kwargs)
    
    # Mostrar resultado en consola
    if args.json_output:
        # Format translations for better readability
        if 'translations' in result:
            result['translations'] = json.loads(json.dumps(result['translations'], indent=2, ensure_ascii=False))
            
        json_result = json.dumps(result, ensure_ascii=False, indent=2)
        audio_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        file_name = f"{audio_name}_{args.tagger_type}_{args.decision_method}.json"
        output_file = os.path.join(tagger.output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_result)
        logger.info(f"Results saved to {output_file}")
    else:
        logger.info("\nResults:")
        logger.info(f"File: {result['file']}")
        if 'transcription' in result:
            logger.info(f"Transcription: {result['transcription']}")
        if 'translations' in result:
            translations_json = json.dumps(result['translations'], indent=2, ensure_ascii=False)
            logger.info(f"Translations:\n{translations_json}")
        logger.info(f"Tag selection method: {args.decision_method}")
        logger.info("Recommended tags:")
        for i, tag_info in enumerate(result['tags']):
            logger.info(f"  {i+1}. {tag_info['tag']} (similarity: {tag_info['similarity']:.4f})")

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 