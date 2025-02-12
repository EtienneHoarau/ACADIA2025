import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Définir le chemin du script en cours d'exécution
script_dir = os.path.dirname(os.path.abspath(__file__))

def random_transform(image):
    """Applique une transformation aléatoire à une image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=random.choice([0, 1, 2, 3]))  # Rotation aléatoire de 0, 90, 180 ou 270 degrés
    
    # Zoom aléatoire
    scale = random.uniform(1, 1.2)
    new_height = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * scale, tf.int32)
    image = tf.image.central_crop(image, central_fraction=0.9)  # Coupe proprement au centre si trop grand
    image = tf.image.resize(image, (256, 256))
    
    # Ajout de bruit gaussien
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    return image

def augment_images(input_dir, output_dir, target_count):
    """Augmente les images d'un dossier pour atteindre un nombre cible d'images."""
    input_dir = os.path.join(script_dir, input_dir)  # Ajuster le chemin d'entrée
    output_dir = os.path.join(script_dir, output_dir)  # Ajuster le chemin de sortie

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_count = len(image_files)
    
    if image_count == 0:
        raise ValueError("Aucune image trouvée dans le dossier d'entrée.")
    
    images = [img_to_array(load_img(os.path.join(input_dir, img))) for img in image_files]
    images = np.array(images) / 255.0  # Normalisation
    
    index = 0
    for i in range(target_count):
        img = images[index % image_count]  # Prend une image originale cycliquement
        img = random_transform(img)  # Applique une transformation
        img = tf.image.convert_image_dtype(img, dtype=tf.uint8)  # Convertit pour l'enregistrement
        save_path = os.path.join(output_dir, f'aug_{i}.png')
        save_img(save_path, img.numpy())
        index += 1
        
    print(f"Génération terminée. {target_count} images ont été enregistrées dans {output_dir}.")

# Exemple d'utilisation
augment_images('dataset_puzzle_base', 'extended_images', 2000)
