import pandas as pd
import numpy as np
import os
import logging
import yake
from .text_embedding_tagger import TextEmbeddingTagger

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRANSCRIPTIONS = [
    "I'm a software engineer with a passion for building scalable and efficient systems. I'm currently working on a project that involves building a new social media platform.",
    "................................. It's made by a music group",
    "Hi, the Spanish restaurant at Clip Point Road is accessible and no stairs.",
    "I am at a little restaurant on Riley Street called Have-2. It used to be Robo-Kog. It's run by two lovely men, Jet and House, who come from Thailand. I'm having a delicious cappuccino. It's a reasonable price restaurant. They do delicious wraps and breakfasts and very nice guys and there is an outside garden at the front of the cafe there is one one step up into the main part of the cafe there's toilets at the back and it's accessible because the guys will help you",
    "I am on the crossroad of the Colón street and heading towards the sea. This crossroad is dangerous because it passes the tram and you can't hear it.",
    "We are at the Burrell road. The Burrell road is totally pedestrian, in principle, totally. No, totally unique platform. And of course, it would be unique if it was a completely pedestrian road, that you could not pass cars, or for example today, which is disabled, like me, or that the pass is restricted, for example, some of the machines, for cars and discards, They do these things but they don't restrict the passenger's ride. And then they also come with the scooters, bicycles and so on. So it's a nice place, but since it's this mixed-race, it's the problem. The road is not pedestrian, it's mixed-race and this is the problem. Anyway, these are things of the Barcelona City Council and many others. Look, now a bike has passed and it's not very slow either.",
    "Això és una trancripció en català per veure si el sistema admet el nostre llenguatge.",
    "El parking de discapacitados del Mercadona no es accesible."
    "Ich bin gerade im Berliner Hauptbahnhof und warte auf meinen Zug nach München. Es ist sehr voll hier und die Anzeigetafel zeigt eine Verspätung von 20 Minuten an. Die Bahnsteige sind barrierefrei zugänglich, aber die Aufzüge sind oft überfüllt.",
    "Una mattina mi son svegliato, o bella ciao, bella ciao, bella ciao ciao ciao! Una mattina mi son svegliato e ho trovato l'invasor. O partigiano portami via, o bella ciao, bella ciao, bella ciao ciao ciao! O partigiano portami via che mi sento di morir. E se io muoio da partigiano, o bella ciao, bella ciao, bella ciao ciao ciao! E se io muoio da partigiano tu mi devi seppellir. E seppellire lassù in montagna, o bella ciao, bella ciao, bella ciao ciao ciao! E seppellire lassù in montagna sotto l'ombra di un bel fior. Tutte le genti che passeranno, o bella ciao, bella ciao, bella ciao ciao ciao! Tutte le genti che passeranno mi diranno «che bel fior!» È questo il fiore del partigiano, o bella ciao, bella ciao, bella ciao ciao ciao! È questo il fiore del partigiano morto per la libertà.",
]   

def get_similarity_tags(tagger_instance, transcription):
    """
    Obtiene el embedding de una transcripción y encuentra los tags más similares.
    """
    # Se pasa un audio_path dummy ya que el embedding se calcula directamente desde la transcripción
    sample_embedding, _ = tagger_instance.get_audio_embedding(audio_path="dummy_audio.wav", transcription=transcription)
    nearest_tags, similarities = tagger_instance.find_similar_tags(sample_embedding)
    return nearest_tags, similarities

def get_semantic_tags(kw_extractor, transcription):
    """
    Kevin's semantic tagger
    """
    logging.info(f"Processing for semantic tags: {transcription[:70]}...")
    keywords = kw_extractor.extract_keywords(transcription)
    logging.info(f"YAKE extracted keywords: {keywords}")
    list_key = [x[0] for x in keywords]
    list_scores = [1-x[1] for x in keywords] # Extract scores
    
    # Limit to top 5 keywords and their scores
    list_key = list_key[:min(5, len(list_key))]
    list_scores = list_scores[:min(5, len(list_scores))]
    
    return list_key, list_scores

def main():
    """
    Función principal para ejecutar el tagger en la lista de transcripciones y guardar los resultados.
    """
    # Ruta al archivo de taxonomía (puedes cambiarlo a 'all_tags.txt' si lo prefieres)
    taxonomy_file = "supertags.txt" 
    
    logging.info(f"Initializing TextEmbeddingTagger with taxonomy: {taxonomy_file}")
    similarity_tagger = TextEmbeddingTagger(taxonomy_file=taxonomy_file)
    logging.info("TextEmbeddingTagger initialized.")
     
    semantic_tagger = yake.KeywordExtractor(n=1, dedupLim=0.9, top=5, features=None)

    results_data = []
    
    # Añadir la fila de encabezado
    header_row = [""] + [f"Tag{i+1}" for i in range(5)] # Asumiendo 5 tags como default k
    results_data.append(header_row)

    for i, transcription in enumerate(TRANSCRIPTIONS):
        logging.info(f"Processing transcription {i+1}/{len(TRANSCRIPTIONS)}: {transcription[:70]}...") # Log de las primeras 70 letras
        nearest_tags, similarities = get_similarity_tags(similarity_tagger, transcription)
        semantic_tags, semantic_scores = get_semantic_tags(semantic_tagger, transcription)

        # Fila de la transcripción
        transcription_row = [transcription] + [""] * 5 # 5 espacios vacíos para las columnas de tags
        results_data.append(transcription_row)
        
        # Fila de resultados similarity tagger (nombres)
        tag_names_row = ["Similarity Tagger"] + nearest_tags[:5] + [""] * (5 - len(nearest_tags)) 
        results_data.append(tag_names_row)
        
        # Fila de similitudes similarity tagger (números)
        similarities_to_add = similarities[:5] + [np.nan] * (5 - len(similarities)) # Usar np.nan para padding numérico
        tag_similarities_row = ["Similarities"] + similarities_to_add
        results_data.append(tag_similarities_row)

        # Fila de resultados semantic tagger (nombres)
        semantic_names_row = ["Semantic Tagger"] + semantic_tags[:5] + [""] * (5 - len(semantic_tags))
        results_data.append(semantic_names_row)
        
        # Fila de resultados semantic tagger (números)
        semantic_scores_to_add = semantic_scores[:5] + [np.nan] * (5 - len(semantic_scores)) # Usar np.nan para padding numérico
        semantic_scores_row = ["Semantic Scores"] + semantic_scores_to_add
        results_data.append(semantic_scores_row)
      
    # Crear DataFrame y guardar a Excel
    df = pd.DataFrame(results_data)
    
    # Establecer la primera fila como encabezado y resetear el índice del DataFrame
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    output_file = "transcription_tagging_results.xlsx"
    
    # Crear un objeto ExcelWriter usando XlsxWriter como motor
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    # Convertir el DataFrame a un objeto Excel de XlsxWriter
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    # Acceder a los objetos workbook y worksheet de XlsxWriter
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Definir un formato de negrita
    bold_format = workbook.add_format({'bold': True})
    
    # Definir formato de gradiente de color (verde para 1, blanco para 0)
    green_gradient_format = {
        'type': '2_color_scale',
        'min_value': 0, 'min_type': 'num', 'min_color': '#FFFFFF', # Blanco
        'max_value': 1, 'max_type': 'num', 'max_color': '#63BE7B'  # Verde
    }

    # Calcular el número de filas de datos por transcripción (5 en este caso)
    rows_per_transcription = 5
    
    # Iterar a través de las transcripciones para aplicar el formato y negritas
    for i in range(len(TRANSCRIPTIONS)):
        # Calcular la fila de Excel donde comienza el bloque de datos de la transcripción actual
        current_block_start_excel_row = 1 + (i * rows_per_transcription) 

        # Aplicar formato de negrita a las etiquetas en la columna A (columna 0)
        # Fila para 'Similarity Tagger'
        similarity_tags_label_excel_row = current_block_start_excel_row + 1 
        worksheet.write(similarity_tags_label_excel_row, 0, 'Similarity Tagger', bold_format)
        
        # Fila para 'Similarities'
        similarities_label_excel_row = current_block_start_excel_row + 2 
        worksheet.write(similarities_label_excel_row, 0, 'Similarities', bold_format)

        # Fila para 'Semantic Tagger'
        semantic_tags_label_excel_row = current_block_start_excel_row + 3 
        worksheet.write(semantic_tags_label_excel_row, 0, 'Semantic Tagger', bold_format)

        # Fila para 'Semantic Scores'
        semantic_scores_label_excel_row = current_block_start_excel_row + 4 
        worksheet.write(semantic_scores_label_excel_row, 0, 'Semantic Scores', bold_format)

        # Aplicar formato condicional a las celdas numéricas (columnas B a F, es decir, 1 a 5)
        # Rango para 'Similarities' (fila numérica)
        similarities_data_excel_row = current_block_start_excel_row + 2
        worksheet.conditional_format(similarities_data_excel_row, 1, similarities_data_excel_row, 5, green_gradient_format)

        # Rango para 'Semantic Scores' (fila numérica)
        semantic_scores_data_excel_row = current_block_start_excel_row + 4
        worksheet.conditional_format(semantic_scores_data_excel_row, 1, semantic_scores_data_excel_row, 5, green_gradient_format)

    # Cerrar el objeto ExcelWriter para guardar el archivo
    writer.close()
    logging.info(f"Results successfully saved to {output_file}")

if __name__ == "__main__":
    main()