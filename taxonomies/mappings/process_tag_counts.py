import os
import json
import csv
import argparse

# Función para cargar el diccionario de mapeo desde el archivo JSON
def load_tag_mapping(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        tag_mapping = json.load(f)
    return tag_mapping

# Función para procesar el archivo CSV con los conteos de tags
def process_tag_counts(tag_mapping, input_file, output_file, unmapped_file):
    # Crear un diccionario para acumular los conteos por categoría
    category_counts = {"": 0}  # Inicializar la categoría vacía con 0
    unmapped_tags = []
    
    # Leer el archivo CSV de entrada
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Saltar la primera fila (encabezados)
        next(csv_reader, None)
        
        for row in csv_reader:
            if len(row) >= 3:  # Asegúrate de que la fila tenga al menos 3 columnas
                tag_id = row[0].strip('"')
                tag_name = row[1].strip('"')
                count = int(row[2].strip('"'))
                
                # Buscar la categoría correspondiente al tag
                category = tag_mapping.get(tag_name, "")
                
                # Acumular el conteo para esta categoría
                if category in category_counts:
                    category_counts[category] += count
                else:
                    category_counts[category] = count
                
                # Si el tag no está mapeado, guardarlo para referencia
                if category == "":
                    unmapped_tags.append((tag_name, count))
    
    # Escribir el archivo CSV de salida con los conteos por categoría
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Category", "Count"])
        
        for category, count in category_counts.items():
            csv_writer.writerow([category, count])
    
    # También guardar un archivo con los tags no mapeados
    with open(unmapped_file, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Tag", "Count"])
        
        for tag, count in unmapped_tags:
            csv_writer.writerow([tag, count])
    
    print(f"Procesamiento completado. Resultados guardados en {output_file}")
    print(f"Tags no mapeados guardados en {unmapped_file}")
    print(f"Total de tags no mapeados: {category_counts['']}")

# Función principal
def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Procesa los conteos de tags utilizando un mapeo predefinido')
    parser.add_argument('--mapping_file', default='16tags_mapping_dict.json', 
                        help='Archivo JSON que contiene el mapeo de tags (predeterminado: 16tags_mapping_dict.json)')
    parser.add_argument('--input_file', default='all_tags_counts.csv',
                        help='Archivo CSV con los conteos de tags originales (predeterminado: all_tags_counts.csv)')
    parser.add_argument('--output_file', default='16tags_counts.csv',
                        help='Archivo CSV de salida para los conteos agrupados (predeterminado: 16tags_counts.csv)')
    parser.add_argument('--unmapped_file', default='16tags_unmapped_tags.csv',
                        help='Archivo CSV para los tags no mapeados (predeterminado 16tags_unmapped_tags.csv)')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    print(f"Cargando diccionario de mapeo de tags desde {args.mapping_file}...")
    tag_mapping = load_tag_mapping(args.mapping_file)
    
    print(f"Se cargaron {len(tag_mapping)} mapeos de tags.")
    
    print("Procesando archivo de conteos de tags...")
    process_tag_counts(tag_mapping, args.input_file, args.output_file, args.unmapped_file)

if __name__ == "__main__":
    main() 