#!/bin/bash
#set -x

### Code to extend the number of images in a dataset using ffmpeg

## To use this script, read the read.me 

# Configuration
INPUT_FOLDER="dataset_puzzle_base_zoomed"  # Folder containing source images
OUTPUT_FOLDER="extended_images"      # Folder for augmented images
NUM_AUGMENTED_IMAGES=2000            # Total number of augmented images to generate
IMAGE_SIZE="512:512"

# Create the output folder if it does not exist
#mkdir -p "$OUTPUT_FOLDER"

echo "Configuration completed."

# Function to add noise
add_noise() {
    local input_file="$1"
    local output_file="$2"
    local intensity=$((RANDOM % 100))
    ffmpeg -i "$input_file" -vf "noise=alls=${intensity}:allf=u,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Function to rotate an image
rotate_image() {
    local input_file="$1"
    local output_file="$2"
    local angle="$3"
    
    case $angle in
        90) transpose="1" ;;
        180) transpose="2" ;;
        270) transpose="3" ;;
        *) echo "Invalid angle"; return 1 ;;
    esac
    
    ffmpeg -i "$input_file" -vf "transpose=$transpose,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Function to flip an image
flip_image() {
    local input_file="$1"
    local output_file="$2"
    local direction="$3"
    
    case $direction in
        horizontal) ffmpeg -i "$input_file" -vf "hflip,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file" ;;
        vertical)   ffmpeg -i "$input_file" -vf "vflip,scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file" ;;
        *) echo "Invalid direction"; return 1 ;;
    esac
}

# Function to zoom an image
zoom_image() {
    local input_file="$1"
    local output_file="$2"
    local zoom_factor="1.1"

    local width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$input_file")
    local height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$input_file")
    
    if [[ -z "$width" || -z "$height" ]]; then
        echo "Error: Unable to retrieve image dimensions."
        return 1
    fi
    
    local zoomed_width=$(awk "BEGIN {print $width / $zoom_factor}")
    local zoomed_height=$(awk "BEGIN {print $height / $zoom_factor}")
    local x_offset=$(awk "BEGIN {srand(); print int(($width - $zoomed_width) * rand())}")
    local y_offset=$(awk "BEGIN {srand(); print int(($height - $zoomed_height) * rand())}")

    ffmpeg -i "$input_file" -vf "crop=${zoomed_width}:${zoomed_height}:${x_offset}:${y_offset},scale=$IMAGE_SIZE" -pix_fmt yuv420p -frames:v 1 -y "$output_file"
}

# Generate augmented images
count=0
image_files=("$INPUT_FOLDER"/*.JPG)

while [ "$count" -lt "$NUM_AUGMENTED_IMAGES" ]; do
    for image_file in "${image_files[@]}"; do
        [ -e "$image_file" ] || continue

        base_name=$(basename "$image_file" | cut -d. -f1)
        nb_copy=$((RANDOM % 20 + 10))  # Increase the number of copies to ensure 2000 images

        for ((i = 0; i < nb_copy && count < NUM_AUGMENTED_IMAGES; i++)); do
            transformation=$((RANDOM % 4))
            output_file="$OUTPUT_FOLDER/${count}.jpg"

            case $transformation in
                0) rotate_image "$image_file" "$output_file" $(( (RANDOM % 3 + 1) * 90 )) ;;
                1) flip_image "$image_file" "$output_file" $([ $((RANDOM % 2)) -eq 0 ] && echo "horizontal" || echo "vertical") ;;
                2) add_noise "$image_file" "$output_file" ;;
                3) zoom_image "$image_file" "$output_file" ;;
            esac

            # Check if the image was successfully created
            if [ -e "$output_file" ]; then
                count=$((count + 1))
            else
                echo "⚠️ Transformation failed, attempt ignored."
            fi

            # Stop if the desired number is reached
            if [ "$count" -ge "$NUM_AUGMENTED_IMAGES" ]; then
                break 2
            fi
        done
    done
done

echo "✅ Augmentation completed: $count images generated in $OUTPUT_FOLDER."
