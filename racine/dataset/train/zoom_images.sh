#!/bin/bash

### Code to zoom images using ffmpeg

## To use this script, read the read.me 

# Folder containing the source images
INPUT_FOLDER="dataset_puzzle_base"
# Folder for the zoomed images
OUTPUT_FOLDER="dataset_puzzle_base_zoomed"
# Zoom factor (1.2 = 20% zoom, 1.5 = 50% zoom)
ZOOM_FACTOR=1.5
# Output size (example: 1024x1024)
IMAGE_SIZE="1024:1024"

# Create the output folder if it doesn't exist
#mkdir -p "$OUTPUT_FOLDER"

# Process each image in the folder
for img in "$INPUT_FOLDER"/*.{jpg,JPG,png,PNG}; do
    [ -e "$img" ] || continue  # Check if the file exists

    filename=$(basename "$img")
    output_file="$OUTPUT_FOLDER/$filename"

    ffmpeg -i "$img" -vf "zoompan=z='min($ZOOM_FACTOR,2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale=$IMAGE_SIZE" -frames:v 1 "$output_file" -y

    echo "Zoomed image saved: $output_file"
done

echo "✅ All images have been zoomed and saved!"
sleep 100s
