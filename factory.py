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
    