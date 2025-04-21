import argparse
import os
import json
from .base_tagger import (
    DECISION_METHOD_KNN,
    DECISION_METHOD_RADIUS,
    DECISION_METHOD_HDBSCAN
)
from .text_embedding_tagger import TextEmbeddingTagger

def create_tagger(tagger_type, taxonomy_file, **kwargs):
    """
    Fábrica para crear el tagger adecuado según el tipo especificado.
    
    Args:
        tagger_type (str): Tipo de tagger ('text', 'audio', 'hybrid')
        taxonomy_file (str): Ruta al archivo de taxonomía
        **kwargs: Argumentos adicionales específicos para cada tipo de tagger
        
    Returns:
        BaseTagger: Instancia del tagger apropiado
    """
    if tagger_type.lower() == 'text':
        return TextEmbeddingTagger(taxonomy_file, **kwargs)
    else:
        raise ValueError(f"Tipo de tagger no reconocido: {tagger_type}. Opciones válidas: 'text'")
    
if __name__ == "__main__":
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Etiquetar un archivo de audio con diferentes modelos")
    
    parser.add_argument("--audio_file", type=str, required=True,
                      help="Ruta al archivo de audio a etiquetar")
    parser.add_argument("--tagger_type", type=str, choices=['text', 'audio', 'hybrid'], default='text',
                      help="Tipo de tagger a utilizar: text (SentenceTransformer), audio (CLAP), hybrid (combinación)")
    parser.add_argument("--taxonomy_file", type=str, default="taxonomies/16tags.txt",
                      help="Ruta al archivo de taxonomía de etiquetas")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Archivo para guardar resultados (CSV o JSON)")
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
                      choices=[DECISION_METHOD_KNN, DECISION_METHOD_RADIUS, DECISION_METHOD_HDBSCAN],
                      help='Método de decisión para seleccionar etiquetas')
    parser.add_argument('--decision_params', type=json.loads, default='{}',
                      help='Parámetros para el método de decisión en formato JSON')
    parser.add_argument("--top_k", type=int, default=None,
                      help="Número de etiquetas a devolver (para KNN)")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Threshhold similaridad para el método Radius Nearest Neighbors")
    parser.add_argument("--min_cluster_size", type=int, default=None,
                      help="Tamaño mínimo de cluster para HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=None,
                      help="Número mínimo de muestras para HDBSCAN")
    
    args = parser.parse_args()

    # Configurar parámetros según el tipo de tagger
    tagger_params = {}
    decision_params = {}
    
    decision_params = {
        'k': args.top_k,
        'threshold': args.threshold, 
        'min_cluster_size': args.min_cluster_size,
        'min_samples': args.min_samples
    }
    # Asignar solo los parámetros que no son None
    decision_params = {k:v for k,v in decision_params.items() if v is not None}

    if args.tagger_type == 'text':
        tagger_params = {
            'model_name': args.text_model,
            'S2TT_model': args.whisper_model,
            'decision_method': args.decision_method,
            'decision_params': decision_params
        }
    elif args.tagger_type == 'audio':
        tagger_params = {
            'model_name': args.clap_model,
            'decision_method': args.decision_method,
            'decision_params': decision_params
        }
    elif args.tagger_type == 'hybrid':
        tagger_params = {
            'audio_model_name': args.audio_model,
            'text_model_name': args.text_model,
            'asr_model_name': args.whisper_model,
            'audio_weight': args.audio_weight,
            'text_weight': args.text_weight,
            'decision_method': args.decision_method,
            'decision_params': decision_params
        }
    
    # Crear tagger
    tagger = create_tagger(args.tagger_type, args.taxonomy_file, **tagger_params)
    
    # Procesar archivo de audio individual
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"El archivo de audio {args.audio_file} no existe")
                
    kwargs = {'language': args.language}
    
    result = tagger.tag_sample(args.audio_file, **kwargs)
    
    # Mostrar resultado en consola
    if args.json_output:
        print("\nResultado:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\nResultado:")
        print(f"Archivo: {result['file']}")
        if 'transcription' in result:
            print(f"Transcripción: {result['transcription']}")
        print(f"Método de selección de etiquetas: {args.decision_method}")
        print("Etiquetas recomendadas:")
        for i, tag_info in enumerate(result['tags']):
            print(f"  {i+1}. {tag_info['tag']} (similitud: {tag_info['similarity']:.4f})") 