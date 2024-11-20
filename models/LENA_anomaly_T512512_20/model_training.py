import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model, load_model
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Charger et préparer les images
def load_and_prepare_image(filepath, img_size=(512, 512)):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0  # Normaliser
    img = np.expand_dims(img, axis=-1)   # Ajouter un canal
    img = np.expand_dims(img, axis=0)    # Ajouter une dimension batch
    return img

# Construire l'encodeur
def build_encoder(input_shape=(512, 512, 1)):
    encoder_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(encoder_input)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 256x256x64
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 128x128x128
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 64x64x256
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 32x32x512
    encoded_output = GlobalAveragePooling2D()(x)  # Sortie : 512
    
    encoder = Model(encoder_input, encoded_output, name="encoder")
    return encoder

# Construire le décodeur
def build_decoder(encoded_dim=512):
    decoder_input = Input(shape=(encoded_dim,))
    x = Dense(32 * 32 * 512, activation="relu")(decoder_input)
    x = Reshape((32, 32, 512))(x)
    x = Conv2DTranspose(256, (3, 3), strides=2, padding="same", activation="relu")(x)  # 64x64x256
    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(x)  # 128x128x128
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)   # 256x256x64
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)   # 512x512x32
    decoded_output = Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)  # 512x512x1
    
    decoder = Model(decoder_input, decoded_output, name="decoder")
    return decoder

# Chemin de sauvegarde du modèle
MODEL_SAVE_PATH = "anomaly_detection_autoencoder.h5"

# Charger l'image d'exemple (pour l'entraînement)
image_path = "Lena_512_base.png"  # Remplacez par le chemin de votre image normale
img = load_and_prepare_image(image_path)

# Vérifier si le modèle existe déjà
if os.path.exists(MODEL_SAVE_PATH):
    print("Modèle trouvé, chargement en cours...")
    autoencoder = load_model(MODEL_SAVE_PATH)
else:
    print("Aucun modèle trouvé, création d'un nouveau modèle...")
    encoder = build_encoder()
    decoder = build_decoder()
    autoencoder = Model(encoder.input, decoder(encoder.output), name="autoencoder")

    # Compiler et entraîner le modèle
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(img, img, epochs=50)  # Ajouter plus d'images pour améliorer l'entraînement

    # Sauvegarder le modèle
    autoencoder.save(MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé sous : {MODEL_SAVE_PATH}")

# Fonction de détection d'anomalie
def detect_anomaly(image, autoencoder, threshold=0.01):
    print("Shape image = ",image.shape)
    reconstructed_image = autoencoder.predict(image)
    mse = np.mean(np.square(image - reconstructed_image))
    return mse, mse > threshold

# Exemple de détection d'anomalie
#test_image_path = "LENA_512_base_anomaly.png"  # Image pour tester la détection
test_image_path = "lena.png"  # Image pour tester la détection
test_img = load_and_prepare_image(test_image_path)
error, is_anomalous = detect_anomaly(test_img, autoencoder)

# Affichage des résultats
print(f"Erreur de reconstruction (MSE): {error}")
if is_anomalous:
    print("Anomalie détectée !")
else:
    print("Pas d'anomalie détectée.")

# Afficher l'image testée et sa reconstruction
reconstructed_img = autoencoder.predict(test_img)

plt.figure(figsize=(10, 5))

# Image d'origine
plt.subplot(1, 3, 1)
print("test image shape = ", test_img.shape)
plt.imshow(test_img[0, :, :, 0], cmap="gray")
plt.title("Image testée")
plt.axis("off")

# Image reconstruite
plt.subplot(1, 3, 2)
print("recons image shape = ", reconstructed_img.shape)
plt.imshow(reconstructed_img[0,:,:,0], cmap="gray")
plt.title("Image reconstruite")
plt.axis("off")

#Erreur
plt.subplot(1,3,3)
plt.imshow(reconstructed_img[0,:,:,0]-test_img[0,:,:,0],cmap="gray")
plt.title("Erreur")
plt.axis("off")

plt.show()
