"""
Tagging Strategy Pattern Implementation

This module provides two strategies for text-to-tags conversion:
- SemanticTaggingStrategy: Uses SentenceTransformer embeddings + KNN/radius matching
- SyntacticTaggingStrategy: Uses YAKE keyword extraction
"""

import os
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Constants for decision methods (reused from base_tagger)
DECISION_METHOD_KNN = "knn"
DEFAULT_K = 5
DECISION_METHOD_RADIUS = "radius"
DEFAULT_RADIUS = 0.4
DECISION_METHOD_ADAPTIVE = "adaptive"
DEFAULT_MIN_THRESHOLD = 0.5

TAGGER_DIR = os.path.dirname(os.path.abspath(__file__))


class TaggingStrategy(ABC):
    """
    Abstract base class for tagging strategies.
    
    All strategies must implement the tag_text method which takes text input
    and returns a list of tags with similarity scores.
    """
    
    @abstractmethod
    def tag_text(self, text: str, top_k: int = 5) -> list:
        """
        Tag the given text and return a list of tags with similarity scores.
        
        Args:
            text (str): The text to tag
            top_k (int): Maximum number of tags to return
            
        Returns:
            list: List of dicts with 'tag' and 'similarity' keys
        """
        pass


class SemanticTaggingStrategy(TaggingStrategy):
    """
    Semantic tagging strategy using SentenceTransformer embeddings.
    
    This strategy encodes the input text into an embedding vector and finds
    the most similar tags from a predefined taxonomy using KNN, radius search,
    or adaptive thresholding.
    """
    
    def __init__(self, taxonomy_file, model_name='paraphrase-multilingual-mpnet-base-v2',
                 device='cpu', decision_method=DECISION_METHOD_KNN, decision_params=None):
        """
        Initialize the semantic tagging strategy.
        
        Args:
            taxonomy_file (str): Name of the taxonomy file (e.g., '16tags.txt')
            model_name (str): SentenceTransformer model name
            device (str): Device to use ('cpu' or 'cuda')
            decision_method (str): Method for tag selection ('knn', 'radius', 'adaptive')
            decision_params (dict): Parameters for the decision method
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self.decision_method = decision_method
        self.tagger_dir = TAGGER_DIR
        self.embeddings_dir = os.path.join(TAGGER_DIR, 'embeddings')
        self.taxonomy_file = os.path.join(TAGGER_DIR, 'taxonomies', taxonomy_file)
        
        # Set up decision parameters with defaults
        self.decision_params = {
            DECISION_METHOD_KNN: {'k': DEFAULT_K},
            DECISION_METHOD_RADIUS: {'threshold': DEFAULT_RADIUS},
            DECISION_METHOD_ADAPTIVE: {'min_threshold': DEFAULT_MIN_THRESHOLD}
        }
        
        # Update with provided parameters
        if decision_params and self.decision_method in decision_params:
            if isinstance(decision_params[self.decision_method], dict):
                self.decision_params[self.decision_method].update(decision_params[self.decision_method])
        
        # Validate decision method
        if decision_method not in [DECISION_METHOD_KNN, DECISION_METHOD_RADIUS, DECISION_METHOD_ADAPTIVE]:
            self.logger.warning(f"Selection method '{decision_method}' not recognized. Using KNN as default.")
            self.decision_method = DECISION_METHOD_KNN
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load embedding model
        start_time = time.time()
        self.logger.info(f"Loading embeddings model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        elapsed = time.time() - start_time
        self.logger.info(f"Embeddings model loaded in {elapsed:.6f}s")
        
        # Load tags
        self.tags = self._load_tags()
        self.logger.info(f"Loaded {len(self.tags)} tags from {taxonomy_file}")
        
        # Load or compute tag embeddings
        self.tag_embeddings = self._load_or_compute_embeddings()
        
        # Initialize decision method
        self._init_decision_method()
    
    def _load_tags(self):
        """Load tags from taxonomy file."""
        start_time = time.time()
        self.logger.info(f"Loading tags from: {self.taxonomy_file}")
        
        with open(self.taxonomy_file, 'r', encoding='utf-8') as f:
            tags = [line.strip().lower() for line in f.readlines() if line.strip()]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tags loaded in {elapsed:.6f}s")
        return tags
    
    def _get_embeddings_file_path(self):
        """Generate the path for embeddings file."""
        taxonomy_name = os.path.splitext(os.path.basename(self.taxonomy_file))[0]
        model_id = f"text_{self.model_name}".replace('/', '_')
        return os.path.join(self.embeddings_dir, f"{taxonomy_name}_{model_id}_embeddings.npz")
    
    def _load_or_compute_embeddings(self):
        """Load existing embeddings or compute them if they don't exist."""
        start_time = time.time()
        embeddings_file = self._get_embeddings_file_path()
        
        if os.path.exists(embeddings_file):
            self.logger.info(f"Loading existing embeddings from {embeddings_file}")
            data = np.load(embeddings_file)
            elapsed = time.time() - start_time
            self.logger.info(f"Embeddings loaded in {elapsed:.6f}s")
            return data['embeddings']
        else:
            self.logger.info(f"Computing embeddings for {len(self.tags)} tags")
            embeddings = []
            for tag in self.tags:
                embedding = self.embedding_model.encode(tag)
                embeddings.append(embedding)
            embeddings = np.array(embeddings)
            
            # Save embeddings
            np.savez(embeddings_file, embeddings=embeddings, tags=np.array(self.tags))
            self.logger.info(f"Embeddings saved to {embeddings_file}")
            
            elapsed = time.time() - start_time
            self.logger.info(f"Embeddings computed and saved in {elapsed:.6f}s")
            return embeddings
    
    def _init_decision_method(self):
        """Initialize the tag selection method."""
        start_time = time.time()
        self.logger.info("Initializing tag selection method")
        
        if self.decision_method == DECISION_METHOD_KNN:
            k = self.decision_params[DECISION_METHOD_KNN]['k']
            self.knn = NearestNeighbors(n_neighbors=min(k, len(self.tags)), metric='cosine')
            self.knn.fit(self.tag_embeddings)
            self.logger.info(f"KNN initialized with k={k}")
        
        elif self.decision_method == DECISION_METHOD_RADIUS:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
            self.rnn = NearestNeighbors(radius=1-threshold, metric='cosine')
            self.rnn.fit(self.tag_embeddings)
            self.logger.info(f"Radius search initialized with threshold={threshold}")
        
        elif self.decision_method == DECISION_METHOD_ADAPTIVE:
            self.knn_all = NearestNeighbors(n_neighbors=len(self.tags), metric='cosine')
            self.knn_all.fit(self.tag_embeddings)
            min_threshold = self.decision_params[DECISION_METHOD_ADAPTIVE]['min_threshold']
            self.logger.info(f"Adaptive method initialized with min_threshold={min_threshold}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tag selection method initialized in {elapsed:.6f}s")
    
    def _find_similar_tags_knn(self, text_embedding, k=None):
        """Find similar tags using KNN."""
        if k is None:
            k = self.decision_params[DECISION_METHOD_KNN]['k']
        
        k = min(k, len(self.tags))
        distances, indices = self.knn.kneighbors(text_embedding.reshape(1, -1), n_neighbors=k)
        
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
        return nearest_tags, similarities
    
    def _find_similar_tags_radius(self, text_embedding):
        """Find similar tags within a radius threshold."""
        threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
        distances, indices = self.rnn.radius_neighbors(text_embedding.reshape(1, -1), radius=1-threshold)
        
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
        # Sort by similarity (highest first)
        sorted_pairs = sorted(zip(nearest_tags, similarities), key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
        return nearest_tags, similarities
    
    def _find_similar_tags_adaptive(self, text_embedding):
        """Find similar tags using adaptive thresholding."""
        min_threshold = self.decision_params[DECISION_METHOD_ADAPTIVE]['min_threshold']
        
        distances, indices = self.knn_all.kneighbors(text_embedding.reshape(1, -1), n_neighbors=len(self.tags))
        
        similarities = [float(1 - distance) for distance in distances[0]]
        tags = [self.tags[idx] for idx in indices[0]]
        
        # Filter tags above threshold
        filtered_pairs = [(tag, sim) for tag, sim in zip(tags, similarities) if sim >= min_threshold]
        
        # If no tags above threshold, return the most similar one
        if not filtered_pairs:
            max_idx = similarities.index(max(similarities))
            return [tags[max_idx]], [similarities[max_idx]]
        
        # Sort by similarity (highest first)
        sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
        return nearest_tags, similarities
    
    def tag_text(self, text: str, top_k: int = 5) -> list:
        """
        Tag text using semantic embeddings.
        
        Args:
            text (str): The text to tag
            top_k (int): Maximum number of tags to return (used for KNN)
            
        Returns:
            list: List of dicts with 'tag' and 'similarity' keys
        """
        start_time = time.time()
        self.logger.info(f"Tagging text using semantic strategy (method: {self.decision_method})")
        
        # Encode the text
        text_embedding = self.embedding_model.encode(text)
        
        # Find similar tags based on decision method
        if self.decision_method == DECISION_METHOD_RADIUS:
            nearest_tags, similarities = self._find_similar_tags_radius(text_embedding)
        elif self.decision_method == DECISION_METHOD_ADAPTIVE:
            nearest_tags, similarities = self._find_similar_tags_adaptive(text_embedding)
        else:  # KNN (default)
            nearest_tags, similarities = self._find_similar_tags_knn(text_embedding, k=top_k)
        
        # Format results
        results = []
        for tag, similarity in zip(nearest_tags, similarities):
            results.append({
                'tag': tag,
                'similarity': similarity
            })
        
        # Limit to top_k for radius and adaptive methods
        if self.decision_method in [DECISION_METHOD_RADIUS, DECISION_METHOD_ADAPTIVE]:
            results = results[:top_k]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Semantic tagging completed in {elapsed:.6f}s, found {len(results)} tags")
        
        return results


class SyntacticTaggingStrategy(TaggingStrategy):
    """
    Syntactic tagging strategy using YAKE keyword extraction.
    
    This strategy extracts keywords from the input text using statistical
    features without requiring any training data or external models.
    """
    
    def __init__(self, language='en', max_ngram_size=1, deduplication_threshold=0.9,
                 num_keywords=5):
        """
        Initialize the syntactic tagging strategy.
        
        Args:
            language (str): Language code for keyword extraction (e.g., 'en', 'es')
            max_ngram_size (int): Maximum size of n-grams to extract
            deduplication_threshold (float): Threshold for deduplicating keywords
            num_keywords (int): Default number of keywords to extract
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.num_keywords = num_keywords
        
        # Import yake here to make it an optional dependency
        try:
            import yake
            self.yake = yake
            self.logger.info(f"YAKE initialized for language: {language}")
        except ImportError:
            raise ImportError("YAKE is required for SyntacticTaggingStrategy. Install it with: pip install yake")
    
    def tag_text(self, text: str, top_k: int = 5) -> list:
        """
        Tag text using YAKE keyword extraction.
        
        Args:
            text (str): The text to tag
            top_k (int): Maximum number of tags to return
            
        Returns:
            list: List of dicts with 'tag' and 'similarity' keys
                  (similarity is always 1.0 for extracted keywords)
        """
        start_time = time.time()
        self.logger.info(f"Tagging text using syntactic strategy (YAKE)")
        
        # Create keyword extractor
        kw_extractor = self.yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            top=top_k,
            features=None
        )
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        
        # Format results (YAKE returns (keyword, score) tuples where lower score = more relevant)
        # We normalize scores to similarity format (higher = better)
        results = []
        if keywords:
            # YAKE scores are lower = better, typically in range [0, 1+]
            # We convert to similarity where 1.0 = most relevant
            max_score = max(score for _, score in keywords) if keywords else 1.0
            
            for keyword, score in keywords[:top_k]:
                # Convert YAKE score to similarity (invert and normalize)
                # Using 1.0 as similarity since YAKE doesn't provide semantic similarity
                similarity = 1.0 - min(score / (max_score + 0.001), 1.0) if max_score > 0 else 1.0
                results.append({
                    'tag': keyword.lower(),
                    'similarity': similarity
                })
        
        elapsed = time.time() - start_time
        self.logger.info(f"Syntactic tagging completed in {elapsed:.6f}s, found {len(results)} keywords")
        
        return results


def create_tagging_strategy(strategy_type: str, taxonomy_file: str = None, **kwargs) -> TaggingStrategy:
    """
    Factory function to create tagging strategies.
    
    Args:
        strategy_type (str): Type of strategy ('semantic' or 'syntactic')
        taxonomy_file (str): Taxonomy file for semantic strategy (required for 'semantic')
        **kwargs: Additional arguments passed to the strategy constructor
        
    Returns:
        TaggingStrategy: The created strategy instance
    """
    logger = logging.getLogger(__name__)
    
    if strategy_type.lower() == 'semantic':
        if taxonomy_file is None:
            raise ValueError("taxonomy_file is required for semantic strategy")
        
        # Extract relevant kwargs for semantic strategy
        model_name = kwargs.get('model_name', 'paraphrase-multilingual-mpnet-base-v2')
        device = kwargs.get('device', 'cpu')
        decision_method = kwargs.get('decision_method', DECISION_METHOD_KNN)
        decision_params = kwargs.get('decision_params', None)
        
        logger.info(f"Creating SemanticTaggingStrategy with model: {model_name}")
        return SemanticTaggingStrategy(
            taxonomy_file=taxonomy_file,
            model_name=model_name,
            device=device,
            decision_method=decision_method,
            decision_params=decision_params
        )
    
    elif strategy_type.lower() == 'syntactic':
        # Extract relevant kwargs for syntactic strategy
        language = kwargs.get('syntactic_language', 'en')
        max_ngram_size = kwargs.get('max_ngram_size', 1)
        deduplication_threshold = kwargs.get('deduplication_threshold', 0.9)
        num_keywords = kwargs.get('num_keywords', 5)
        
        logger.info(f"Creating SyntacticTaggingStrategy for language: {language}")
        return SyntacticTaggingStrategy(
            language=language,
            max_ngram_size=max_ngram_size,
            deduplication_threshold=deduplication_threshold,
            num_keywords=num_keywords
        )
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Valid options: 'semantic', 'syntactic'")

