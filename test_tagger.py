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

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories for taxonomies and consolidated output relative to BASE_DIR
TAXONOMIES_DIR = os.path.join(BASE_DIR, "taxonomies")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def test_tagger_on_folder(path_to_folder, taxonomy_filename,
                           embedding_model="paraphrase-multilingual-mpnet-base-v2",
                           whisper_model="openai/whisper-small", language=None, top_k=5):
    """
    Ejecuta el etiquetador en todos los archivos de audio de una carpeta para una taxonomía específica.
    Devuelve un DataFrame con los resultados.

    Args:
        path_to_folder (str): Ruta a la carpeta que contiene los archivos de audio.
        taxonomy_filename (str): Nombre del archivo de taxonomía (e.g., "supertags.txt").
        embedding_model (str): Modelo de embeddings a utilizar.
        whisper_model (str): Modelo de Whisper a utilizar para ASR.
        language (str): Idioma para la transcripción.
        top_k (int): Número de etiquetas a devolver.

    Returns:
        pandas.DataFrame: DataFrame con los resultados para la taxonomía.
    """
    print(f"\nProcesando archivos de audio para taxonomía: {taxonomy_filename}")
    taxonomy_file_path = os.path.join(TAXONOMIES_DIR, taxonomy_filename)

    # Get list of audio files in the folder
    audio_files = [f for f in os.listdir(path_to_folder) if f.endswith('.wav')]

    if not audio_files:
        print(f"No se encontraron archivos de audio en {path_to_folder}")
        return None

    print(f"Encontrados {len(audio_files)} archivos de audio")

    # Get duration of each audio file
    audio_durations = {}
    for audio_file in audio_files:
        try:
            audio_path = os.path.join(path_to_folder, audio_file)
            duration = librosa.get_duration(path=audio_path)
            audio_durations[audio_file] = duration
        except Exception as e:
            print(f"Error al obtener duración de {audio_file}: {str(e)}")
            audio_durations[audio_file] = 0

    # Initialize the tagger
    tagger = TextEmbeddingTagger(
        taxonomy_file=taxonomy_file_path, # Use full path here
        model_name=embedding_model,
        S2TT_model=whisper_model
    )

    # Create index for rows for the output DataFrame
    row_names = ['rtfx', 'transcription_orig', 'transcription_eng']
    for i in range(top_k):
        row_names.append(f'tag_{i+1}')
        row_names.append(f'similarity_{i+1}')

    # Create dictionary to store results for this taxonomy
    taxonomy_results_data = {}

    # Process each audio file
    for audio_file in tqdm(audio_files, desc=f"Procesando {taxonomy_filename}"):
        # Initialize list for current audio file's data, filled with placeholders
        current_audio_data = [''] * len(row_names) # Default empty string or appropriate placeholder

        try:
            audio_path = os.path.join(path_to_folder, audio_file)
            start_time = time.time()
            result = tagger.tag_sample(audio_path, language=language, k=top_k)
            end_time = time.time()
            process_time = end_time - start_time
            audio_duration = audio_durations.get(audio_file, 0)
            rtfx = audio_duration / process_time if process_time > 0 else 0

            # Populate current_audio_data based on row_names order
            current_audio_data[row_names.index('rtfx')] = rtfx
            current_audio_data[row_names.index('transcription_orig')] = result['transcription']
            #current_audio_data[row_names.index('transcription_eng')] = result['transcription_eng']

            for i, tag_info in enumerate(result['tags']):
                if i < top_k:
                    tag_row_name = f'tag_{i+1}'
                    sim_row_name = f'similarity_{i+1}'
                    # Ensure the index exists before assigning
                    if tag_row_name in row_names:
                        current_audio_data[row_names.index(tag_row_name)] = tag_info['tag']
                    if sim_row_name in row_names:
                        current_audio_data[row_names.index(sim_row_name)] = tag_info['similarity']

            taxonomy_results_data[audio_file] = current_audio_data

        except Exception as e:
            print(f"Error procesando {audio_file} para {taxonomy_filename}: {str(e)}")
            # Fill with error values for the current audio file
            # Only fill the transcription_orig with error message, others remain as initialized
            current_audio_data[row_names.index('transcription_orig')] = f"ERROR: {str(e)}"
            taxonomy_results_data[audio_file] = current_audio_data
    
    # Convert dictionary to DataFrame. Transpose it to have audio files as columns and row_names as index.
    df = pd.DataFrame(taxonomy_results_data, index=row_names)
    return df

def run_all_taxonomy_tests(path_to_audio_folder, embedding_model, whisper_model, language=None, top_k=5):
    """
    Ejecuta el etiquetador en todas las taxonomías disponibles y consolida los resultados.

    Args:
        path_to_audio_folder (str): Ruta a la carpeta que contiene los archivos de audio.
        embedding_model (str): Modelo de embeddings a utilizar.
        whisper_model (str): Modelo de Whisper a utilizar para ASR.
        language (str): Idioma para la transcripción.
        top_k (int): Número de etiquetas a devolver.
    """
    print(f"Iniciando evaluación para todas las taxonomías en {TAXONOMIES_DIR}")

    taxonomy_files = [f for f in os.listdir(TAXONOMIES_DIR) if f.endswith('.txt') and os.path.isfile(os.path.join(TAXONOMIES_DIR, f))]
    if not taxonomy_files:
        print(f"No se encontraron archivos de taxonomía (.txt) en {TAXONOMIES_DIR}")
        return

    all_results_dfs = []

    for taxonomy_filename in taxonomy_files:
        taxonomy_name = os.path.splitext(taxonomy_filename)[0] # e.g., "supertags"
        df_taxonomy = test_tagger_on_folder(
            path_to_audio_folder,
            taxonomy_filename,
            embedding_model=embedding_model,
            whisper_model=whisper_model,
            language=language,
            top_k=top_k
        )
        if df_taxonomy is not None:
            # Rename columns to include taxonomy name prefix
            df_taxonomy.columns = [f"{taxonomy_name}_{col}" for col in df_taxonomy.columns]
            all_results_dfs.append(df_taxonomy)

    if not all_results_dfs:
        print("No se generaron resultados para ninguna taxonomía.")
        return

    # Concatenate all DataFrames horizontally. They should align by their common index (row_names).
    final_df = pd.concat(all_results_dfs, axis=1)

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    output_file = os.path.join(OUTPUT_DIR, 'consolidated_tagging_results_og.xlsx')
    final_df.to_excel(output_file, float_format='%.4f')
    print(f"\nTodos los resultados consolidados en: {output_file}")


if __name__ == "__main__":
    # Define parameters for the test
    AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "../audios_test")
    EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
    WHISPER_MODEL = "openai/whisper-small"
    LANGUAGE = None # Auto-detect
    TOP_K_TAGS = 5

    # Run the consolidated test
    run_all_taxonomy_tests(
        path_to_audio_folder=AUDIO_FOLDER_PATH,
        embedding_model=EMBEDDING_MODEL,
        whisper_model=WHISPER_MODEL,
        language=LANGUAGE,
        top_k=TOP_K_TAGS
    )
