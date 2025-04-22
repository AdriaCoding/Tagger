import os
import json
import glob
import argparse

# Función para leer los archivos JSON y extraer los tags
def create_tag_mapping(mapping_name):
    tag_mapping = {}
    
    # Obtener todos los archivos JSON en el directorio de mapping
    json_files = glob.glob(os.path.join( mapping_name, "*.json"))
    
    # Procesar cada archivo JSON
    for json_file in json_files:
        # Obtener el nombre de la categoría (nombre del archivo sin extensión)
        category = os.path.basename(json_file).replace(".json", "")
        
        # Leer el archivo JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Buscar la sección de datos que contiene los tags
                for item in data:
                    if isinstance(item, dict) and "type" in item and item["type"] == "table" and "data" in item:
                        # Extraer los nombres de los tags
                        for tag_data in item["data"]:
                            if "name" in tag_data:
                                tag_name = tag_data["name"]
                                # Añadir al diccionario de mapeo
                                tag_mapping[tag_name] = category
            except json.JSONDecodeError:
                print(f"Error al leer el archivo JSON: {json_file}")
    
    return tag_mapping

# Función principal
def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Crea un diccionario de mapeo de tags')
    parser.add_argument('--mapping_name', default='16tags', 
                        help='Nombre del directorio que contiene los archivos JSON de mapeo (predeterminado: 16tags)')
    
    # Parsear argumentos
    args = parser.parse_args()
    mapping_name = args.mapping_name
    
    print(f"Creando diccionario de mapeo de tags para '{mapping_name}'...")
    tag_mapping = create_tag_mapping(mapping_name)
    
    print(f"Se encontraron {len(tag_mapping)} tags mapeados a categorías.")
    
    # Guardar en JSON
    json_file = os.path.join("", f"{mapping_name}_mapping_dict.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(tag_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Diccionario de mapeo guardado en formato JSON: {json_file}")

if __name__ == "__main__":
    main() 