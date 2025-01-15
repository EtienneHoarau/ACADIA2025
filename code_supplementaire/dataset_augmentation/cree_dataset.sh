#!/bin/bash

# Configuration
INPUT_FOLDER="good"          # Dossier contenant les images source
OUTPUT_FOLDER="./augmented_images"     # Dossier pour les images augmentées
NUM_AUGMENTED_IMAGES=10000              # Nombre total d'images augmentées à générer

# Créer le dossier de sortie s'il n'existe pas
#mkdir -p "$OUTPUT_FOLDER"

echo "config terminé"

# Fonction pour ajouter du grain uniforme à une image
add_noise() {
    local input_file="$1"
    local output_file="$2"
    local nb_inten=$((RANDOM % 100))
    local intensity="${3:-$nb_inten}" # Intensité du bruit, par défaut 30

    # Ajouter un grain uniforme à l'image
    ffmpeg -i "$input_file" -vf "noise=alls=${intensity}:allf=u" -y "$output_file"
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
    esac
    ffmpeg -i "$input_file" -vf "transpose=$transpose" -y "$output_file"
}

# Fonction pour flipper une image
flip_image() {
    local input_file="$1"
    local output_file="$2"
    local direction="$3"
    if [ "$direction" = "horizontal" ]; then
        ffmpeg -i "$input_file" -vf "hflip" -y "$output_file"
    elif [ "$direction" = "vertical" ]; then
        ffmpeg -i "$input_file" -vf "vflip" -y "$output_file"
    fi
}

# Fonction pour zoomer une image avec un centre aléatoire cohérent
zoom_image() {
    local input_file="$1"
    local output_file="$2"
    local zoom_factor="$3"

    echo "Préparation du zoom"

    # Obtenir la largeur et la hauteur de l'image d'origine
    local width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$input_file")
    local height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$input_file")

    echo "Dimensions récupérées : largeur = ${width}, hauteur = ${height}"

    if [[ -z "$width" || -z "$height" ]]; then
        echo "Erreur : Impossible de récupérer les dimensions de l'image." >&2
        return 1
    fi

    # Calcul des dimensions de la zone zoomée avec awk pour gérer les nombres à virgule
    local zoomed_width=$(awk "BEGIN {print $width / $zoom_factor}")
    local zoomed_height=$(awk "BEGIN {print $height / $zoom_factor}")

    echo "Dimensions zoomées calculées : largeur = ${zoomed_width}, hauteur = ${zoomed_height}"

    # Vérification pour éviter des dépassements (si zoom_factor > 1)
    if (( $(awk "BEGIN {print ($zoom_factor > 1)}") )); then
        # Calcul des coordonnées aléatoires pour le centre de la zone zoomée
        local x_offset=$(awk "BEGIN {srand(); print int(($width - $zoomed_width) * rand())}")
        local y_offset=$(awk "BEGIN {srand(); print int(($height - $zoomed_height) * rand())}")
    else
        # Pas de zoom, on reste sur l'image entière
        local x_offset=0
        local y_offset=0
    fi

    echo "Coordonnées de la zone zoomée : x_offset = ${x_offset}, y_offset = ${y_offset}"

    # Appliquer le zoom et recentrer sur les coordonnées aléatoires
    ffmpeg -i "$input_file" -vf "crop=${zoomed_width}:${zoomed_height}:${x_offset}:${y_offset},scale=${width}:${height}" -y "$output_file"
}

# Compteur pour les images générées
count=0
#while[ "$(ls -1 "$OUTPUT_FOLDER" | wc -l)" -lt 2000 ]; do
    # Parcourir toutes les images dans le dossier source
    for image_file in "$INPUT_FOLDER"/*.{jpg,jpeg,png}; do
        # Assurez-vous que le fichier existe
        [ -e "$image_file" ] || continue
        #echo "début boucle"
        # Nom de base de l'image
        base_name=$(basename "$image_file")
        base_name="${base_name%.*}"
        nb_copy=$((RANDOM%20))
        sub_count=0
        name_img=$image_file
        #echo "image a modifié = ${name_img}"

        # Appliquer des transformations jusqu'à atteindre le nombre souhaité
        while [ $sub_count -lt $nb_copy ]; do
            #echo "début transfo : ${count}"
            nb_transfo=$((RANDOM % 3))
            count_transfo=0
            #applique aléatoirement 1,2 ou 3 transfo à l'image
            #while [ $count_transfo -lt $nb_transfo ]; do
                # Choisir une transformation aléatoire
                transformation=$((RANDOM % 4))
                echo "transformation prête"
                case $transformation in
                    0)  # Rotation
                        angle=$(( (RANDOM % 3 + 1) * 90 ))
                        rotate_image "$name_img" "$OUTPUT_FOLDER/${base_name}$count.jpg" $angle
                        ;;
                    1)  # Flip
                        direction=$([ $((RANDOM % 2)) -eq 0 ] && echo "horizontal" || echo "vertical")
                        flip_image "$name_img" "$OUTPUT_FOLDER/${base_name}$count.jpg" $direction
                        ;;
                    2)  # Bruit
                        add_noise "$name_img" "$OUTPUT_FOLDER/${base_name}$count.jpg"
                        ;;
                     3)  # Zoom ou dézoom
                        zoom_factor=$(awk "BEGIN { print 1 + (rand()  * 0.5 ) }")  # Facteur de zoom entre 0.8 et 1.2
                        zoom_factor=1.1
                        echo "zoom = ${zoom_factor}"
                        zoom_image "$name_img" "$OUTPUT_FOLDER/${base_name}$count.jpg" "$zoom_factor"
                        ;;    
                esac
                #echo "transo faite"
                count_transfo=$((count_transfo + 1))
                name_img=$OUTPUT_FOLDER/${base_name}$count.jpg
                #echo "nom nouvelle image ${name_img}"
            #done
            # Incrémenter le compteur
            count=$((count + 1))
            sub_count=$((sub_count+1))
            if [ $count -ge $NUM_AUGMENTED_IMAGES ]; then
                break
            fi
        done

        # Vérifier si on a atteint le nombre total
        if [ $count -ge $NUM_AUGMENTED_IMAGES ]; then
            break
        fi
    done
#done

echo "Augmentation terminée : $count images générées dans le dossier $OUTPUT_FOLDER."
sleep 20s