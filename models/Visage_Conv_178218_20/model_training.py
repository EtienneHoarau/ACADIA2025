import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


# Paramètres de base
image_size = (208, 208)
data_dir = "C:\\Users\\hoara\\Documents\\COURS\\5A\\PLP\\datasets\\img_align_celeba"  # Dossier de visages
checkpoint_filepath = 'autoencoder_faces'
encoder_filepath = "./encoder/" + checkpoint_filepath + "_encoder"
decoder_filepath = "./decoder/" + checkpoint_filepath + "_decoder"

IMG_HEIGHT, IMG_WIDTH = image_size
BATCH_SIZE = 8
EPOCHS = 20
NB_IMAGE = 500

# Chargement et prétraitement des images
def load_images(data_dir, image_size, n_images=NB_IMAGE, color_mode='rgb'):
    images = []
    image_paths = os.listdir(data_dir)[:n_images]
    for img_file in image_paths:
        img = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_COLOR if color_mode == 'rgb' else cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (image_size[1], image_size[0]))  # Ajustement de taille pour (178, 218)
            img = img.astype('float32') / 255.0  # Normalisation
            images.append(img)
    images = np.array(images)
    if color_mode == 'rgb':
        return images.reshape(-1, image_size[0], image_size[1], 3)  # Reshape pour Conv2D (RGB)
    else:
        return images.reshape(-1, image_size, image_size, 1)  # Reshape pour Conv2D (Grayscale)

# Charger les images et les redimensionner en 178x218 pour le modèle
X_train = load_images(data_dir, image_size, color_mode='rgb')
X_test = X_train[:int(NB_IMAGE/2)]
X_train = X_train[int(NB_IMAGE/2):]

# Vérifier si le modèle existe déjà
if not os.path.exists(checkpoint_filepath):
    print("Pas de modèle de sauvegarde trouvé. Création d'un nouveau modèle.")

    # Définition de l'encodeur
    encoder_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(encoder_input)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    encoded_output = layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    print("Output shape = ", encoded_output.shape[1:])
    encoder = models.Model(encoder_input, encoded_output, name="encoder")

    # Définition du décodeur
    decoder_input = layers.Input(shape=encoded_output.shape[1:])
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu")(decoder_input)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    decoded_output = layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid")(x)

    decoder = models.Model(decoder_input, decoded_output, name="decoder")

    print("Forme crée")
    # Autoencodeur complet
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    encoder.summary()
    decoder.summary()
    
    # Adaptation pour correspondre aux dimensions du décodeur
    print("Adaptation")
    #reshaped_encoded = layers.Reshape((IMG_HEIGHT // 16, IMG_WIDTH // 16, 256))(encoded)
    autoencoder_output = decoder(encoded)
    autoencoder = models.Model(inputs=autoencoder_input, outputs=autoencoder_output)

    # Compilation du modèle
    print("Compilation")
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics="accuracy")
    autoencoder.summary()
    # Entraîner le modèle
    autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

    # Sauvegarder le modèle complet et les sous-modèles
    autoencoder.save(checkpoint_filepath, save_format='tf')
    encoder.save(encoder_filepath, save_format='tf')
    decoder.save(decoder_filepath, save_format='tf')
    print("Modèles sauvegardés.")
else:
    print("Chargement des modèles existants.")
    autoencoder = tf.keras.models.load_model(checkpoint_filepath)
    encoder = tf.keras.models.load_model(encoder_filepath)
    decoder = tf.keras.models.load_model(decoder_filepath)

# Fonction pour afficher des reconstructions
def plot_reconstructions(model, images, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig, axes = plt.subplots(2, n_images, figsize=(15, 5))
    for i in range(n_images):
        # Image originale
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")

        # Image reconstruite
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis("off")
    plt.show()

# Afficher des exemples de reconstructions pour évaluer la performance
plot_reconstructions(autoencoder, X_train)
