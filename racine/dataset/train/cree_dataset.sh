#!/bin/bash
#set -x

# Configuration
INPUT_FOLDER="dataset_puzzle_base_zoomed"  # Dossier contenant les images source
OUTPUT_FOLDER="extended_images"      # Dossier pour les images augmentées
NUM_AUGMENTED_IMAGES=2000            # Nombre total d'images augmentées à générer
IMAGE_SIZE="512:512"

# Créer le dossier de sortie s'il n'existe pas
#mkdir -p "$OUTPUT_FOLDER"

echo "Configuration terminée."

# Fonction pour ajouter du bruit
add_noise() {
    local input_file="$1"
    local output_file="$2"
    local intensity=$((RANDOM % 100))
    ffmpeg -i "$input_file" -vf "noise=alls=${intensity}:allf=u,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Fonction pour tourner une image
rotate_image() {
    local input_file="$1"
    local output_file="$2"
    local angle="$3"
    
    case $angle in
        90) transpose="1" ;;
        180) transpose="2" ;;
        270) transpose="3" ;;
        *) echo "Angle non valide"; return 1 ;;
    esac
    
    ffmpeg -i "$input_file" -vf "transpose=$transpose,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Fonction pour retourner une image
flip_image() {
    local input_file="$1"
    local output_file="$2"
    local direction="$3"
    
    case $direction in
        horizontal) ffmpeg -i "$input_file" -vf "hflip,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file" ;;
        vertical)   ffmpeg -i "$input_file" -vf "vflip,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file" ;;
        *) echo "Direction non valide"; return 1 ;;
    esac
}

# Fonction pour zoomer une image
zoom_image() {
    local input_file="$1"
    local output_file="$2"
    local zoom_factor="1.1"

    local width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$input_file")
    local height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$input_file")
    
    if [[ -z "$width" || -z "$height" ]]; then
        echo "Erreur : Impossible de récupérer les dimensions de l'image."
        return 1
    fi
    
    local zoomed_width=$(awk "BEGIN {print $width / $zoom_factor}")
    local zoomed_height=$(awk "BEGIN {print $height / $zoom_factor}")
    local x_offset=$(awk "BEGIN {srand(); print int(($width - $zoomed_width) * rand())}")
    local y_offset=$(awk "BEGIN {srand(); print int(($height - $zoomed_height) * rand())}")

    ffmpeg -i "$input_file" -vf "crop=${zoomed_width}:${zoomed_height}:${x_offset}:${y_offset},scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Génération des images augmentées
count=0
image_files=("$INPUT_FOLDER"/*.JPG)

while [ "$count" -lt "$NUM_AUGMENTED_IMAGES" ]; do
    for image_file in "${image_files[@]}"; do
        [ -e "$image_file" ] || continue

        base_name=$(basename "$image_file" | cut -d. -f1)
        nb_copy=$((RANDOM % 20 + 10))  # Augmenter le nombre de copies pour garantir 2000 images

        for ((i = 0; i < nb_copy && count < NUM_AUGMENTED_IMAGES; i++)); do
            transformation=$((RANDOM % 4))
            output_file="$OUTPUT_FOLDER/${count}.jpg"

            case $transformation in
                0) rotate_image "$image_file" "$output_file" $(( (RANDOM % 3 + 1) * 90 )) ;;
                1) flip_image "$image_file" "$output_file" $([ $((RANDOM % 2)) -eq 0 ] && echo "horizontal" || echo "vertical") ;;
                2) add_noise "$image_file" "$output_file" ;;
                3) zoom_image "$image_file" "$output_file" ;;
            esac

            # Vérifier que l'image a bien été créée
            if [ -e "$output_file" ]; then
                count=$((count + 1))
            else
                echo "⚠️ Échec de la transformation, tentative ignorée."
            fi

            # Arrêter si on a atteint le nombre voulu
            if [ "$count" -ge "$NUM_AUGMENTED_IMAGES" ]; then
                break 2
            fi
        done
    done
done

echo "✅ Augmentation terminée : $count images générées dans $OUTPUT_FOLDER."
