# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:01:21 2024

@author: Kraline
"""
import os
import numpy as np
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
data_dir = "..\\train\\puzzle_extend"  # Dossier de visages

# Chemin complet où sauvegarder le fichier
directory = "."
NB_IMAGE = 400
filename = "convdata"+str(NB_IMAGE)+".npy"
test = os.path.join(directory, filename) #Mettre le nom du fichier voulu dans le répertoire voulu
image_size = (256,256)

IMG_HEIGHT, IMG_WIDTH = image_size
def load_images(data_dir, image_size, n_images=NB_IMAGE, color_mode='rgb'):
    images = []
    image_paths = os.listdir(data_dir)[:n_images]
    for img_file in image_paths:
        img = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_COLOR if color_mode == 'rgb' else cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if color_mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size[1], image_size[0]))  # Ajustement de taille pour (512, 512)
            img = img.astype('float32') / 255.0  # Normalisation
            images.append(img)

    images = np.array(images)
    print(images.shape)
    if color_mode == 'rgb':
        images=images.reshape(-1, image_size[0], image_size[1], 3)  # Reshape pour Conv2D (RGB)
        np.save(test, images)
        return images
    else:
        images=images.reshape(-1, image_size[0], image_size[1], 1)  # Reshape pour Conv2D (Grayscale)
        np.save(test, images)
    # Sauvegarder dans un fichier .npy


modifieddata = load_images(data_dir, image_size, color_mode='rgb')
