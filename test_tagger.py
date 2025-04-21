import pandas as pd
import os
import time
from tqdm import tqdm
import warnings
import sys
import librosa
from .text_embedding_tagger import TextEmbeddingTagger

# Suprimir todos los warnings
warnings.filterwarnings("ignore")

def test_tagger_on_folder(path_to_folder, taxonomy_file="taxonomies/16tags.txt", 
                           embedding_model="paraphrase-multilingual-mpnet-base-v2", 
                           whisper_model="openai/whisper-small", language="en", top_k=5):
    """
    Ejecuta el etiquetador en todos los archivos de audio de una carpeta y guarda los resultados en un CSV.
    
    Args:
        path_to_folder (str): Ruta a la carpeta que contiene los archivos de audio.
        taxonomy_file (str): Ruta al archivo de taxonomía de etiquetas.
        embedding_model (str): Modelo de embeddings a utilizar.
        whisper_model (str): Modelo de Whisper a utilizar para ASR.
        language (str): Idioma para la transcripción.
        top_k (int): Número de etiquetas a devolver.
    
    Returns:
        pandas.DataFrame: DataFrame con los resultados.
    """
    print(f"Procesando archivos de audio en: {path_to_folder}")
    print(f"Usando taxonomía: {taxonomy_file}")
    print(f"Modelo ASR: {whisper_model}")
    print(f"Modelo de embeddings: {embedding_model}")
    
    # Obtener lista de archivos de audio en la carpeta
    audio_files = [f for f in os.listdir(path_to_folder) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"No se encontraron archivos de audio en {path_to_folder}")
        return None
    
    print(f"Encontrados {len(audio_files)} archivos de audio")
    
    # Obtener duración de cada archivo de audio
    audio_durations = {}
    for audio_file in audio_files:
        try:
            audio_path = os.path.join(path_to_folder, audio_file)
            duration = librosa.get_duration(path=audio_path)
            audio_durations[audio_file] = duration
            print(f"Duración de {audio_file}: {duration:.2f} segundos")
        except Exception as e:
            print(f"Error al obtener duración de {audio_file}: {str(e)}")
            audio_durations[audio_file] = 0
    
    # Inicializar el etiquetador
    tagger = TextEmbeddingTagger(
        taxonomy_file=taxonomy_file,
        model_name=embedding_model,
        whisper_model=whisper_model
    )
    
    # Crear diccionario para almacenar resultados
    results_dict = {}
    
    # Procesar cada archivo de audio
    for audio_file in tqdm(audio_files, desc="Procesando archivos"):
        # Crear columnas para este archivo
        col_text = f"{audio_file}_text"
        col_scores = f"{audio_file}_scores"
        
        try:
            # Ruta completa al archivo de audio
            audio_path = os.path.join(path_to_folder, audio_file)
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Etiquetar audio
            result = tagger.tag_audio(audio_path, language=language, k=top_k)
            
            end_time = time.time()
            process_time = end_time - start_time
            
            # Calcular RTFx (tiempo_audio / tiempo_proceso)
            audio_duration = audio_durations.get(audio_file, 0)
            rtfx = audio_duration / process_time if process_time > 0 else 0
            
            # Inicializar listas para esta columna
            col_text_values = [result['transcription']]
            col_scores_values = [rtfx]
            
            # Añadir tags y similitudes
            for tag_info in result['tags']:
                col_text_values.append(tag_info['tag'])
                col_scores_values.append(tag_info['similarity'])
            
            # Guardar en el diccionario de resultados
            results_dict[col_text] = col_text_values
            results_dict[col_scores] = col_scores_values
            
            print(f"  - {audio_file} procesado en {process_time:.2f} segundos (RTFx: {rtfx:.2f})")
            
        except Exception as e:
            print(f"Error procesando {audio_file}: {str(e)}")
            
            # Inicializar con valores por defecto en caso de error
            col_text_values = [f"ERROR: {str(e)}"]
            col_scores_values = [0.0]
            
            # Rellenar con valores vacíos para los tags y similitudes
            for i in range(top_k):
                col_text_values.append("")
                col_scores_values.append(0.0)
            
            # Guardar en el diccionario de resultados
            results_dict[col_text] = col_text_values
            results_dict[col_scores] = col_scores_values
    
    # Crear índice para filas
    row_names = ['transcription']
    for i in range(top_k):
        row_names.append(f'tag_{i+1}')
    
    # Convertir diccionario a DataFrame
    df = pd.DataFrame(results_dict, index=row_names)
    
    # Guardar resultados
    output_file = os.path.join(path_to_folder, 'tagging_results.csv')
    df.to_csv(output_file, float_format='%.4f')
    print(f"\nResultados guardados en: {output_file}")
    
    return df

if __name__ == "__main__":
    # Ejecutar el etiquetador en la carpeta de audios de prueba
    test_tagger_on_folder("audios_test")
