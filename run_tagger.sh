#!/usr/bin/env bash
# run_tagger.sh: lanza el Tagger con el preload/threads hack para el servidor de megafone.net

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
export OMP_NUM_THREADS=1

# Configurar directorio de caché de transformers a uno con permisos adecuados
export TRANSFORMERS_CACHE=/srv/www/blind.wiki/public_html/Tagger/cache
export HF_HOME=/srv/www/blind.wiki/public_html/Tagger/cache

# activa tu virtualenv
source /srv/www/blind.wiki/public_html/.venv/bin/activate

# lanza el Tagger con parámetros
# $1: audio_file
# $2: output_json_path
# Resto de argumentos ($3 en adelante) se pueden usar para otros parámetros de Tagger.main si es necesario
python -m Tagger.main \
  --audio_file "$1" \
  --output_json_path "$2" \
  --device cpu \
  --language en \
  --decision_method adaptive \
  --taxonomy_file supertags.txt \
  $3 $4 $5 $6 $7 $8 $9
