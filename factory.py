"""
Factory module for creating taggers and tagging strategies.

This module provides factory functions to create:
- Tagger instances (TextEmbeddingTagger)
- Tagging strategies (Semantic or Syntactic)
"""

from .text_embedding_tagger import TextEmbeddingTagger
from .tagging_strategy import (
    TaggingStrategy,
    SemanticTaggingStrategy,
    SyntacticTaggingStrategy,
    create_tagging_strategy
)


def create_tagger(tagger_type, taxonomy_file, tagging_strategy='semantic', **kwargs):
    """
    Fábrica para crear el tagger adecuado según el tipo especificado.
    
    Args:
        tagger_type (str): Tipo de tagger ('text', 'audio', 'hybrid')
        taxonomy_file (str): Ruta al archivo de taxonomía
        tagging_strategy (str): Estrategia de etiquetado ('semantic' o 'syntactic')
        **kwargs: Argumentos adicionales específicos para cada tipo de tagger
            For semantic strategy:
                - model_name (str): SentenceTransformer model name
                - decision_method (str): 'knn', 'radius', or 'adaptive'
                - decision_params (dict): Parameters for decision method
            For syntactic strategy:
                - syntactic_language (str): Language code for YAKE (e.g., 'en', 'es')
                - max_ngram_size (int): Maximum n-gram size
                - deduplication_threshold (float): Deduplication threshold
                - num_keywords (int): Number of keywords to extract
        
    Returns:
        BaseTagger: Instancia del tagger apropiado
        
    Example:
        # Create tagger with semantic strategy (default)
        tagger = create_tagger('text', '16tags.txt')
        
        # Create tagger with syntactic strategy
        tagger = create_tagger('text', '16tags.txt', tagging_strategy='syntactic')
        
        # Create tagger with semantic strategy and custom parameters
        tagger = create_tagger('text', '16tags.txt', 
                               tagging_strategy='semantic',
                               decision_method='adaptive',
                               decision_params={'adaptive': {'min_threshold': 0.6}})
    """
    if tagger_type.lower() == 'text':
        # Extract strategy-specific kwargs
        strategy_kwargs = {
            'syntactic_language': kwargs.pop('syntactic_language', 'en'),
            'max_ngram_size': kwargs.pop('max_ngram_size', 1),
            'deduplication_threshold': kwargs.pop('deduplication_threshold', 0.9),
            'num_keywords': kwargs.pop('num_keywords', 5),
        }
        
        return TextEmbeddingTagger(
            taxonomy_file,
            tagging_strategy_type=tagging_strategy,
            **strategy_kwargs,
            **kwargs
        )
    else:
        raise ValueError(f"Tipo de tagger no reconocido: {tagger_type}. Opciones válidas: 'text'")
