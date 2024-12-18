import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Paramètres de base
checkpoint_filepath = 'autoencoder_faces_512_12'
encoder_filepath = "./encoder/" + checkpoint_filepath + "_encoder"
decoder_filepath = "./decoder/" + checkpoint_filepath + "_decoder"

image_size = (256,256)
IMG_HEIGHT, IMG_WIDTH = image_size
BATCH_SIZE = 32
EPOCHS = 10
NB_IMAGE = 500


#PSNR
def calculate_psnr(img1, img2):
    # Vérifier si les images ont les mêmes dimensions
    if img1.shape != img2.shape:
        raise ValueError("Les images doivent avoir les mêmes dimensions pour calculer le PSNR.")

    # Calculer l'erreur quadratique moyenne (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Les images sont identiques

    # Calculer le PSNR
    max_pixel = 1  # Valeur maximale pour les images en 8 bits
    psnr = 10 * np.log10((max_pixel**2) / np.sqrt(mse))
    return psnr
## Charger les images et les diviser en ensembles d'entraînement et de test
#
X_train = np.load("..\..\..\\Dataset\\treatedimage\\convisage500.npy", allow_pickle=True)#directory du fichier image prétraiter
X_test = X_train[:int(NB_IMAGE/5)]
X_train = X_train[int(NB_IMAGE/5):]

im_test=X_test[1]
plt.imshow(im_test)

# Vérifier si le modèle existe déjà
if not os.path.exists(checkpoint_filepath):
    print("Pas de modèle de sauvegarde trouvé. Création d'un nouveau modèle.")

    # Définition de l'encodeur
    encoder_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(3, kernel_size=(4, 4), strides=2, padding="same")(encoder_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Deuxième couche
    x = Conv2D(32, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Troisième couche
    x = Conv2D(32, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Quatrième couche
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Cinquième couche
    x = Conv2D(64, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Sixième couche
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Septième couche
    x = Conv2D(128, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Huitième couche
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Neuvième couche
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

   # Dernière couche
    x = Conv2D(500, kernel_size=(8, 8), strides=1, padding="valid")(x)
    encoded_output = BatchNormalization()(x)

    encoded_output = Conv2D(512, kernel_size=(4, 4), strides=2, padding="same")(x)
    encoder = models.Model(encoder_input, encoded_output, name="encoder")

    # Définition du décodeur
    decoder_input = layers.Input(shape=encoded_output.shape[1:])
    x = Conv2DTranspose(32, kernel_size=(8, 8), strides=1, padding="valid")(decoder_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Deuxième couche
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Troisième couche
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Quatrième couche
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Cinquième couche
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Sixième couche
    x = Conv2DTranspose(32, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Septième couche
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Huitième couche
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Neuvième couche
    x = Conv2DTranspose(32, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Dixième couche
    x = Conv2DTranspose(32, kernel_size=(4, 4), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # # Onzième couche
    # x = Conv2DTranspose(3, kernel_size=(4, 4), strides=2, padding="same")(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    # Dernière couche - Sigmoid pour la reconstruction
    decoded_output = Conv2DTranspose(3, kernel_size=(4, 4), strides=2, padding="same", activation="sigmoid")(x)
    decoder = models.Model(decoder_input, decoded_output, name="decoder")

    # Autoencodeur complet
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    autoencoder_output = decoder(encoded)
    autoencoder = models.Model(inputs=autoencoder_input, outputs=autoencoder_output)

    # Compilation du modèle
    autoencoder.compile(optimizer=Nadam(learning_rate=0.001), loss="mse", metrics=["accuracy"])

    # Résumé des modèles
    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    # Entraîner le modèle
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10,min_delta=1e-4,verbose=0,mode='auto')
    history = autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, X_test), shuffle=True)

    # Récupérer les informations   
    nbperiode=np.arange(1,21,1)
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    nb_periode=np.arange(1,EPOCHS+1)
   
    #Affichage des courbe
    # Premier graphe
    plt.subplot(2, 2, 1)  # 2 lignes, 1 colonne, 1er sous-graphe
    plt.plot(nb_periode, train_accuracy, color="blue")
    plt.title("Précision de l'entrainement")
    plt.grid(True)
    # Deuxième graphe
    plt.subplot(2, 2, 2)  # 2 lignes, 1 colonne, 2e sous-graphe
    plt.plot(nb_periode,val_accuracy, color="red")
    plt.title("Précision de la validation")
    plt.grid(True)
    #
    plt.subplot(2, 2, 3)  # 2 lignes, 1 colonne, 2e sous-graphe
    plt.plot(nb_periode,train_loss, color="purple")
    plt.title("Loss de l'entrainement")
    plt.grid(True)
    #
    plt.subplot(2, 2, 4)  # 2 lignes, 1 colonne, 2e sous-graphe
    plt.plot(nb_periode, val_loss, color="orange")
    plt.title("Loss de la validation")
    plt.grid(True)
    # Afficher les sous-graphes
    plt.tight_layout()  # Ajuste les marges pour éviter les chevauchements
    plt.show()

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

# Fonction pour visualiser les reconstructions pour test model
def plot_reconstructions(model, images, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    print(images.shape)
    fig = plt.figure(figsize=(n_images * 1.5, 3))

    for image_index in range(n_images):
        # Image d'origine
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")

        # Image reconstruite
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")
        print("1 = ",str(calculate_psnr(images[image_index], reconstructions[image_index])))
   
    plt.show()

# Afficher des exemples de reconstructions pour évaluer la performance
plot_reconstructions(autoencoder, X_test)



# Fonction Chargement et évaluation des anomalies
def load_and_prepare_image(filepath, img_size):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img.reshape(-1, image_size[0], image_size[1], 3)

def plot_anomaly(model, image):
    reconstruction = model.predict(image)
    erreur = image - reconstruction
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title("Image originale")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(reconstruction[0])
    plt.title("Reconstruction")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(erreur[0]) + 0.5)
    plt.title("Erreur")
    plt.axis("off")
    plt.show()

# Test de reconstruction et visualisation des anomalies
# erreur_dir = "D:\\cours\\PLP\\dataset\\img_align_celeba\\head2.jpg"
# erreur = load_and_prepare_image(erreur_dir, image_size)
# plot_anomaly(autoencoder, erreur)
