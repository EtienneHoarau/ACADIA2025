import os
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

base_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, os.pardir, "train", "puzzle_extend")
save_dir = os.path.join(base_dir, os.pardir, "train", "puzzle_numpy")
NB_IMAGE = 2000
image_size_number = 512
filename = f"convdata_{image_size_number}_{NB_IMAGE}.npy"
file_path = os.path.join(save_dir, filename)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Chemin absolu des données :", data_dir)
print("Chemin absolu de sauvegarde :", save_dir)
print("Fichier de sortie :", file_path)

# Vérifier et créer le dossier de sauvegarde
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

test = os.path.join(save_dir, filename)

image_size = (image_size_number, image_size_number)

# Allocation d'un fichier mémoire-mappé pour stocker les images
shape = (NB_IMAGE, image_size[0], image_size[1], 3)
dtype = np.float32  # Normalisation entre 0 et 1

memmap_array = np.memmap(test, dtype=dtype, mode='w+', shape=shape)

def load_images_to_memmap(data_dir, memmap_array, image_size, color_mode='rgb'):
    image_paths = os.listdir(data_dir)[:NB_IMAGE]
    for i, img_file in enumerate(image_paths):
        img = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_COLOR if color_mode == 'rgb' else cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if color_mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size[1], image_size[0]))  # Redimensionnement
            img = img.astype('float32') / 255.0  # Normalisation

            # Sauvegarde directe dans le fichier mémoire-mappé
            memmap_array[i] = img

    print(f"Images sauvegardées dans {test}")

load_images_to_memmap(data_dir, memmap_array, image_size, color_mode='rgb')

# Synchronisation des modifications sur le disque
memmap_array.flush()