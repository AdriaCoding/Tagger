import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import hdbscan


# Definir constantes para métodos de selección de etiquetas
DECISION_METHOD_KNN = "knn"
DEFAULT_K = 5
DECISION_METHOD_RADIUS = "radius"
DEFAULT_RADIUS = 0.5
DECISION_METHOD_HDBSCAN = "hdbscan"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 1
TAGGER_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(TAGGER_DIR, 'embeddings')

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
            taxonomy_file (str): Ruta al archivo de taxonomía
            device (str): Dispositivo a utilizar (cuda o cpu)
            decision_method (str): Método para seleccionar etiquetas ('knn', 'radius', 'hdbscan')
            decision_params (dict): Parámetros adicionales para el método de selección:
                - Para 'knn': {'k': número de vecinos (default: 5)}
                - Para 'radius': {'threshold': threshhold de similaridad mínimo (default: 0.5)}
                - Para 'hdbscan': {'min_cluster_size': tamaño mínimo de cluster (default: 5),
                                  'min_samples': muestras mínimas (default: 1)}
        """
        self.taxonomy_file = taxonomy_file
        self.embeddings_dir = EMBEDDINGS_DIR
        self.decision_method = decision_method
        
        # Configurar parámetros por defecto si no se especifican
        self.decision_params = {
            DECISION_METHOD_KNN: {'k': DEFAULT_K},
            DECISION_METHOD_RADIUS: {'threshold': DEFAULT_RADIUS},
            DECISION_METHOD_HDBSCAN: {'min_cluster_size': DEFAULT_MIN_CLUSTER_SIZE, 'min_samples': DEFAULT_MIN_SAMPLES}
        }
        
        # Actualizar con los parámetros proporcionados
        if decision_params:
            if self.decision_method in self.decision_params and isinstance(decision_params, dict):
                self.decision_params[self.decision_method].update(decision_params)
            
        # Validar el método de selección de etiquetas
        if decision_method not in [DECISION_METHOD_KNN, DECISION_METHOD_RADIUS, DECISION_METHOD_HDBSCAN]:
            print(f"ADVERTENCIA: Método de selección '{decision_method}' no reconocido. Usando KNN por defecto.")
            self.decision_method = DECISION_METHOD_KNN
               
        # Configurar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Usando dispositivo: {self.device}")
        
        # Crear directorio de embeddings si no existe
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Cargar etiquetas
        self.tags = self.load_tags(taxonomy_file)
        print(f"Se cargaron {len(self.tags)} etiquetas desde {taxonomy_file}")
        
        # Cargar o calcular embeddings de etiquetas
        self.tag_embeddings = self.load_or_compute_embeddings()
        
        # Inicializar el método de selección de etiquetas apropiado
        self._init_decision_method()
        
        print(f"Método de selección de etiquetas: {self.decision_method}")
    
    def _init_decision_method(self):
        """
        Inicializa el método de selección de etiquetas según la configuración.
        """
        if self.decision_method == DECISION_METHOD_KNN:
            k = self.decision_params[DECISION_METHOD_KNN]['k']
            self.knn = NearestNeighbors(n_neighbors=min(k, len(self.tags)), metric='cosine')
            self.knn.fit(self.tag_embeddings)
        
        elif self.decision_method == DECISION_METHOD_RADIUS:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
            self.rnn = NearestNeighbors(radius=1-threshold, metric='cosine')
            self.rnn.fit(self.tag_embeddings)
        
        elif self.decision_method == DECISION_METHOD_HDBSCAN:
            # No es necesario inicializar HDBSCAN aquí, se creará en cada consulta
            pass
    
    def load_tags(self, taxonomy_file):
        """
        Carga y preprocesa las etiquetas desde un archivo de taxonomía.
        
        Args:
            taxonomy_file (str): Ruta al archivo de taxonomía
            
        Returns:
            list: Lista de etiquetas procesadas
        """
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            tags = [line.strip().lower() for line in f.readlines()]
        
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
        embeddings = []
        print("Calculando embeddings para etiquetas...")
        for tag in tqdm(self.tags):
            embedding = self.get_tag_embedding(tag)
            embeddings.append(embedding)
        
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
        embeddings_file = self.get_embeddings_file_path()
        
        # Verificar si el archivo de embeddings existe
        if os.path.exists(embeddings_file):
            print(f"Cargando embeddings existentes desde {embeddings_file}")
            data = np.load(embeddings_file)
            return data['embeddings']
        else:
            print(f"Calculando embeddings para {len(self.tags)} etiquetas...")
            embeddings = self.compute_tag_embeddings()
            
            # Guardar embeddings
            np.savez(embeddings_file, embeddings=embeddings, tags=np.array(self.tags))
            print(f"Embeddings guardados en {embeddings_file}")
            
            return embeddings
    
    def find_similar_tags_knn(self, sample_embedding, k=None):
        """
        Encuentra las etiquetas más similares usando K-Nearest Neighbors.
        
        Args:
            sample_embedding: Embedding de la muestra
            k (int): Número de etiquetas a devolver, anula el k por defecto
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        if k is None:
            k = self.decision_params[DECISION_METHOD_KNN]['k']
        
        # Asegurar que k no sea mayor que el número de etiquetas disponibles
        k = min(k, len(self.tags))
        
        # Encontrar etiquetas más cercanas
        distances, indices = self.knn.kneighbors(sample_embedding.reshape(1, -1), n_neighbors=k)
        print(distances)
        print(indices)
        # Obtener etiquetas y calcular similitudes
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
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
        if threshold is None:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
        
        # Encontrar etiquetas dentro del radio
        distances, indices = self.rnn.radius_neighbors(sample_embedding.reshape(1, -1), radius=1-threshold)
        
        # Obtener etiquetas y calcular similitudes
        nearest_tags = [self.tags[idx] for idx in indices[0]]
        similarities = [float(1 - distance) for distance in distances[0]]
        
        # Ordenar por similitud (mayor a menor)
        sorted_pairs = sorted(zip(nearest_tags, similarities), key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
        return nearest_tags, similarities
    
    def find_similar_tags_hdbscan(self, sample_embedding, min_cluster_size=None, min_samples=None):
        """
        Encuentra las etiquetas más similares usando HDBSCAN clustering.
        
        Args:
            sample_embedding: Embedding de la muestra
            min_cluster_size (int): Tamaño mínimo del cluster
            min_samples (int): Número mínimo de muestras
            
        Returns:
            tuple: (lista de etiquetas similares, lista de similitudes)
        """
        # Crear un conjunto de datos que incluya tanto el embedding de la muestra como los embeddings de las etiquetas
        combined_embeddings = np.vstack([sample_embedding.reshape(1, -1), self.tag_embeddings])
        
        # Ejecutar HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(combined_embeddings)
        
        # Obtener el cluster de la muestra
        sample_cluster = cluster_labels[0]
        
        # Si la muestra es ruido (cluster -1), usar KNN como fallback
        if sample_cluster == -1:
            print("ADVERTENCIA: La muestra no pertenece a ningún cluster (es ruido).")
        
        # Encontrar todos los embeddings de etiquetas que pertenecen al mismo cluster
        cluster_indices = np.where(cluster_labels[1:] == sample_cluster)[0]
        
        if len(cluster_indices) == 0:
            print("ADVERTENCIA: No hay etiquetas en el mismo cluster que la muestra.")
        
        # Calcular distancias y similitudes desde el embedding de la muestra a los embeddings de las etiquetas del cluster
        distances = []
        for idx in cluster_indices:
            distance = np.linalg.norm(sample_embedding - self.tag_embeddings[idx])
            distances.append(distance)
        
        # Normalizar distancias a [0, 1]
        if len(distances) > 0:
            max_distance = max(distances)
            if max_distance > 0:
                normalized_distances = [d / max_distance for d in distances]
            else:
                normalized_distances = [0.0] * len(distances)
        else:
            normalized_distances = []
        
        # Calcular similitudes
        similarities = [1.0 - d for d in normalized_distances]
        
        # Obtener etiquetas y ordenar por similitud
        nearest_tags = [self.tags[idx] for idx in cluster_indices]
        
        # Ordenar etiquetas por similitud
        sorted_pairs = sorted(zip(nearest_tags, similarities), key=lambda x: x[1], reverse=True)
        nearest_tags = [tag for tag, _ in sorted_pairs]
        similarities = [sim for _, sim in sorted_pairs]
        
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
        if self.decision_method == DECISION_METHOD_RADIUS:
            threshold = self.decision_params[DECISION_METHOD_RADIUS]['threshold']
            return self.find_similar_tags_radius(sample_embedding, threshold)
        elif self.decision_method == DECISION_METHOD_HDBSCAN:
            min_cluster_size = self.decision_params[DECISION_METHOD_HDBSCAN]['min_cluster_size']
            min_samples = self.decision_params[DECISION_METHOD_HDBSCAN]['min_samples']
            return self.find_similar_tags_hdbscan(sample_embedding, min_cluster_size, min_samples)
        else:  # Fallback to KNN
            k = self.decision_params[DECISION_METHOD_KNN]['k']
            return self.find_similar_tags_knn(sample_embedding, k)
    
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