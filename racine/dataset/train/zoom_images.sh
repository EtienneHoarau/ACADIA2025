#!/bin/bash

# Dossier contenant les images source
INPUT_FOLDER="dataset_puzzle_base"
# Dossier pour les images zoomées
OUTPUT_FOLDER="dataset_puzzle_base_zoomed"
# Facteur de zoom (1.2 = zoom 20%, 1.5 = zoom 50%)
ZOOM_FACTOR=1.5
# Taille de sortie (exemple : 1024x1024)
IMAGE_SIZE="1024:1024"

# Créer le dossier de sortie s'il n'existe pas
#mkdir -p "$OUTPUT_FOLDER"

# Traitement de chaque image du dossier
for img in "$INPUT_FOLDER"/*.{jpg,JPG,png,PNG}; do
    [ -e "$img" ] || continue  # Vérifie si le fichier existe

    filename=$(basename "$img")
    output_file="$OUTPUT_FOLDER/$filename"

    ffmpeg -i "$img" -vf "zoompan=z='min($ZOOM_FACTOR,2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale=$IMAGE_SIZE" -frames:v 1 "$output_file" -y

    echo "Image zoomée enregistrée : $output_file"
done

echo "✅ Toutes les images ont été zoomées et enregistrées !"
sleep 100s
