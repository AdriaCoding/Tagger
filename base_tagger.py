import numpy as np
import os
import torch
import logging
import time
import json
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
# import hdbscan


# Definir constantes para métodos de selección de etiquetas
DECISION_METHOD_KNN = "knn"
DEFAULT_K = 5
DECISION_METHOD_RADIUS = "radius"
DEFAULT_RADIUS = 0.4
# DECISION_METHOD_HDBSCAN = "hdbscan"
# DEFAULT_MIN_CLUSTER_SIZE = 5
# DEFAULT_MIN_SAMPLES = 1
DECISION_METHOD_ADAPTIVE = "adaptive"
DEFAULT_MIN_THRESHOLD = 0.5
TAGGER_DIR = os.path.dirname(os.path.abspath(__file__))

class BaseTagger(ABC):
    """
    Clase base abstracta para los diferentes etiquetadores
    Define la interfaz común para todos los tipos de etiquetadores
    """
    
    def __init__(self, taxonomy_file, device=None, 
                decision_method=DECISION_METHOD_KNN, decision_params=None):
        """
        Inicialización común para todos los etiquetadores
        
        Args:
            taxonomy_file (str): Nombre del archivo de taxonomía
            device (str): Dispositivo a utilizar (cuda o cpu)
            decision_method (str): Método para seleccionar etiquetas ('knn', 'radius', 'hdbscan')
            decision_params (dict): Parámetros adicionales para el método de selección:
                - Para 'knn': {'k': número de vecinos (default: 5)}
                - Para 'radius': {'threshold': threshhold de similaridad mínimo (default: 0.5)}
                - Para 'hdbscan': {'min_cluster_size': tamaño mínimo de cluster (default: 5),
                                  'min_samples': muestras mínimas (default: 1)}
        """
        self.logger = logging.getLogger(__name__)
        start_time = time.time()
        self.logger.info("Initializing base tagger")
        
        self.tagger_dir = TAGGER_DIR
        self.output_dir = os.path.join(TAGGER_DIR, 'output')
        self.taxonomy_file = os.path.join(TAGGER_DIR, 'taxonomies', taxonomy_file)
        self.embeddings_dir = os.path.join(TAGGER_DIR, 'embeddings')
        self.decision_method = decision_method
        
        # Configurar parámetros por defecto si no se especifican
        self.decision_params = {
            DECISION_METHOD_KNN: {'k': DEFAULT_K},
            DECISION_METHOD_RADIUS: {'threshold': DEFAULT_RADIUS},
            # DECISION_METHOD_HDBSCAN: {'min_cluster_size': DEFAULT_MIN_CLUSTER_SIZE, 'min_samples': DEFAULT_MIN_SAMPLES},
            DECISION_METHOD_ADAPTIVE: {'min_threshold': DEFAULT_MIN_THRESHOLD}
        }
        
        # Actualizar con los parámetros proporcionados
        if decision_params and self.decision_method in decision_params:
            if isinstance(decision_params[self.decision_method], dict):
                self.decision_params[self.decision_method].update(decision_params[self.decision_method])
            
        # Validar el método de selección de etiquetas
        if decision_method not in [DECISION_METHOD_KNN, DECISION_METHOD_RADIUS, DECISION_METHOD_ADAPTIVE]:  # DECISION_METHOD_HDBSCAN removed
            self.logger.warning(f"Selection method '{decision_method}' not recognized. Using KNN as default.")
            self.decision_method = DECISION_METHOD_KNN
               
        # Configurar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Crear directorio de embeddings si no existe
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Cargar etiquetas
        self.tags = self.load_tags(self.taxonomy_file)
        self.logger.info(f"Loaded {len(self.tags)} tags from {taxonomy_file}")
        
        # Cargar o calcular embeddings de etiquetas
        self.tag_embeddings = self.load_or_compute_embeddings()
        
        # Inicializar el método de selección de etiquetas apropiado
        self._init_decision_method()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Base tagger initialization completed in {elapsed:.2f}s")
        self.logger.info(f"Tag selection method: {self.decision_method}")
    
    def _init_decision_method(self):
        """
        Inicializa el método de selección de etiquetas según la configuración.
        """
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
            # Para el método adaptativo, necesitaremos usar KNN para seleccionar todas las etiquetas
            self.knn_all = NearestNeighbors(n_neighbors=len(self.tags), metric='cosine')
            self.knn_all.fit(self.tag_embeddings)
            min_threshold = self.decision_params[DECISION_METHOD_ADAPTIVE]['min_threshold']
            self.logger.info(f"Adaptive method initialized with min_threshold={min_threshold}")
            
        elapsed = time.time() - start_time
        self.logger.info(f"Tag selection method initialized in {elapsed:.2f}s")
    
    def load_tags(self, taxonomy_file):
        """
        Carga y preprocesa las etiquetas desde un archivo de taxonomía.
        
        Args:
            taxonomy_file (str): Ruta al archivo de taxonomía
            
        Returns:
            list: Lista de etiquetas procesadas
        """
        start_time = time.time()
        self.logger.info(f"Loading tags from: {taxonomy_file}")
        
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            tags = [line.strip().lower() for line in f.readlines()]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tags loaded in {elapsed:.2f}s")
        return tags
    
    @abstractmethod
    def get_audio_embedding(self, sample_path):
        """
        Obtiene el embedding para un audio, ya sea directamente o a partir de una transcripción.
        Debe ser implementado por las subclases.
        
        Args:
            sample_path: Ruta a la muestra
            
        Returns:
            numpy.ndarray: Vector de embedding
            string: Transcripción (opcional)
        """
        pass
    
    @abstractmethod
    def get_tag_embedding(self, tag):
        """
        Obtiene el embedding para una etiqueta.
        Debe ser implementado por las subclases.
        
        Args:
            tag: Texto de la etiqueta
            
        Returns:
            numpy.ndarray: Vector de embedding
        """
        pass
    
    def compute_tag_embeddings(self):
        """
        Calcula embeddings para todas las etiquetas.
        
        Returns:
            numpy.ndarray: Matriz de embeddings
        """
        start_time = time.time()
        self.logger.info("Computing embeddings for all tags")
        
        embeddings = []
        for tag in tqdm(self.tags, desc="Computing tag embeddings"):
            embedding = self.get_tag_embedding(tag)
            embeddings.append(embedding)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tag embeddings computed in {elapsed:.2f}s")
        return np.array(embeddings)
    
    def get_embeddings_file_path(self):
        """
        Genera la ruta del archivo para guardar/cargar los embeddings.
        
        Returns:
            str: Ruta del archivo
        """
        taxonomy_name = os.path.splitext(os.path.basename(self.taxonomy_file))[0]
        model_id = self.get_model_identifier().replace('/', '_')
        return os.path.join(self.embeddings_dir, f"{taxonomy_name}_{model_id}_embeddings.npz")
    
    @abstractmethod
    def get_model_identifier(self):
        """
        Devuelve un identificador único para el modelo.
        Debe ser implementado por las subclases.
        
        Returns:
            str: Identificador del modelo
        """
        pass
    
    def load_or_compute_embeddings(self):
        """
        Carga embeddings existentes o los calcula si no existen.
        
        Returns:
            numpy.ndarray: Matriz de embeddings de etiquetas
        """
        start_time = time.time()
        embeddings_file = self.get_embeddings_file_path()
        
        # Verificar si el archivo de embeddings existe
        if os.path.exists(embeddings_file):
            self.logger.info(f"Loading existing embeddings from {embeddings_file}")
            data = np.load(embeddings_file)
            elapsed = time.time() - start_time
            self.logger.info(f"Embeddings loaded in {elapsed:.2f}s")
            return data['embeddings']
        else:
            self.logger.info(f"Computing embeddings for {len(self.tags)} tags")
            embeddings = self.compute_tag_embeddings()
            
            # Guardar embeddings
            np.savez(embeddings_file, embeddings=embeddings, tags=np.array(self.tags))
            self.logger.info(f"Embeddings saved to {embeddings_file}")
            
            elapsed = time.time() - start_time
            self.logger.info(f"Embeddings computed and saved in {elapsed:.2f}s")
            return embeddings
    
    def find_similar_tags_knn(self, sample_embedding):
        """
        Encuentra las etiquetas más similares usando K-Nearest Neighbors.
        
        Args:
            sample_embedding: Embedding de la muestra
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        start_time = time.time()
        k = self.decision_params[DECISION_METHOD_KNN]['k']
        
        self.logger.info(f"Finding {k} nearest neighbors")
        
        # Asegurar que k no sea mayor que el número de etiquetas disponibles
        k = min(k, len(self.tags))
        
        # Encontrar etiquetas más cercanas
        distances, indices = self.knn.kneighbors(sample_embedding.reshape(1, -1), n_neighbors=k)
        # Obtener etiquetas y calcular similitudes
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
        elapsed = time.time() - start_time
        self.logger.info(f"KNN search completed in {elapsed:.2f}s")
        self.logger.debug(f"Found tags: {json.dumps(list(zip(nearest_tags, similarities)), indent=2)}")
        
        return nearest_tags, similarities
    
    def find_similar_tags_radius(self, sample_embedding, threshold=None):
        """
        Encuentra las etiquetas más similares dentro de un radio determinado.
        
        Args:
            sample_embedding: Embedding de la muestra
            radius (float): Threshhold de similaridad mínimo (1-radius)
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        start_time = time.time()
        if threshold is None:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
        
        self.logger.info(f"Finding neighbors within threshold {threshold}")
        
        # Encontrar etiquetas dentro del radio
        distances, indices = self.rnn.radius_neighbors(sample_embedding.reshape(1, -1), radius=1-threshold)
        
        # Obtener etiquetas y calcular similitudes
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
        # Ordenar por similitud (mayor a menor)
        sorted_pairs = sorted(zip(nearest_tags, similarities), key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Radius search completed in {elapsed:.2f}s")
        self.logger.debug(f"Found tags: {json.dumps(list(zip(nearest_tags, similarities)), indent=2)}")
        
        return nearest_tags, similarities
    
    def find_similar_tags_adaptive(self, sample_embedding, min_threshold=None):
        """
        Encuentra las etiquetas más similares con similaridad > threshold.
        Si no hay ninguna etiqueta por encima del threshold, devuelve la más similar.
        
        Args:
            sample_embedding: Embedding de la muestra
            min_threshold (float): Threshold mínimo de similaridad (por defecto 0.5)
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        start_time = time.time()
        if min_threshold is None:
            min_threshold = self.decision_params[DECISION_METHOD_ADAPTIVE]['min_threshold']
        
        self.logger.info(f"Finding tags with adaptive threshold {min_threshold}")
        
        # Obtener todas las etiquetas y sus similitudes
        distances, indices = self.knn_all.kneighbors(sample_embedding.reshape(1, -1), n_neighbors=len(self.tags))
        
        # Calcular similitudes (1 - distancia)
        similarities = [float(1 - distance) for distance in distances[0]]
        tags = [self.tags[idx] for idx in indices[0]]
        
        # Filtrar aquellas por encima del threshold
        filtered_pairs = [(tag, sim) for tag, sim in zip(tags, similarities) if sim >= min_threshold]
        
        # Si no hay ninguna por encima del threshold, tomar la más similar
        if not filtered_pairs:
            # Encontrar la etiqueta con mayor similitud
            max_idx = similarities.index(max(similarities))
            self.logger.info("No tags above threshold, using most similar tag")
            return [tags[max_idx]], [similarities[max_idx]]
        
        # Ordenar por similitud (mayor a menor)
        sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Adaptive search completed in {elapsed:.2f}s")
        self.logger.debug(f"Found tags: {json.dumps(list(zip(nearest_tags, similarities)), indent=2)}")
        
        return nearest_tags, similarities
    
    def find_similar_tags(self, sample_embedding):
        """
        Encuentra las etiquetas más similares a una muestra según el método configurado.
        Los parametros del método usados son aquellos de la configuracion de clase, self.decision_params
        
        Args:
            sample_embedding: Embedding de la muestra
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        start_time = time.time()
        self.logger.info(f"Finding similar tags using method: {self.decision_method}")
        
        if self.decision_method == DECISION_METHOD_RADIUS:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
            result = self.find_similar_tags_radius(sample_embedding, threshold)
        # elif self.decision_method == DECISION_METHOD_HDBSCAN:
        #     min_cluster_size = self.decision_params[DECISION_METHOD_HDBSCAN]['min_cluster_size']
        #     min_samples = self.decision_params[DECISION_METHOD_HDBSCAN]['min_samples']
        #     return self.find_similar_tags_hdbscan(sample_embedding, min_cluster_size, min_samples)
        elif self.decision_method == DECISION_METHOD_ADAPTIVE:
            min_threshold = self.decision_params[DECISION_METHOD_ADAPTIVE]['min_threshold']
            result = self.find_similar_tags_adaptive(sample_embedding, min_threshold)
        else:  # Fallback to KNN
            result = self.find_similar_tags_knn(sample_embedding)
            
        elapsed = time.time() - start_time
        self.logger.info(f"Tag search completed in {elapsed:.2f}s")
        return result
    
    def tag_sample(self, sample_path, **kwargs):
        """
        Etiqueta una muestra.
        
        Args:
            sample_path (str): Ruta a la muestra
            **kwargs: Argumentos adicionales específicos del modelo
            
        Returns:
            dict: Diccionario con resultados
        """
        start_time = time.time()
        self.logger.info(f"Starting sample tagging: {sample_path}")
        
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
            
        elapsed = time.time() - start_time
        self.logger.info(f"Sample tagging completed in {elapsed:.2f}s")
        self.logger.debug(f"Tagging result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result