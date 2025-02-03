# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:15:20 2025

@author: Charles Dijon and Etienne Hoarau 
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Base parameters
image_size = (256,256)
NB_IMAGE = 1000
version=1
data_dir = "..\\dataset\\convert_dataset\\convdata"+str(NB_IMAGE)+".npy"  # pretreated file
checkpoint_filepath = '..\\model_trained\\model_V'+str(version)
erreur_dir = "..\\dataset\\test\\sratch\\noir.JPG" #Nature of the anomaly
erreur_dir2 = "..\\dataset\\test\\cut\\coupé.JPG"

IMG_HEIGHT, IMG_WIDTH = image_size
BATCH_SIZE = 8
EPOCHS = 50
learn_rate=0.001 #Base 0.001
lossfunction="mse"
validation_data_split=0.2 #pourcentage (ex:0.2)
patienceval=5 #tolerance of validation loss degradation

X_train = np.load(data_dir, allow_pickle=True)
X_test = X_train[:6]
X_train = X_train[6:]

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
   
    plt.show()

# Chargement et évaluation des anomalies
def load_and_prepare_image(filepath, img_size):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img.reshape(-1, image_size[0], image_size[1], 3)

def plot_anomaly(model, image):
    reconstruction = model.predict(image)
    erreur = image[:,:,:,1] - reconstruction[:,:,:,1]
    print("SHape of error ",erreur.shape)
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
    return [reconstruction,erreur]


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

    # Dernière couche - Sigmoid pour la reconstruction
    decoded_output = Conv2DTranspose(3, kernel_size=(4, 4), strides=2, padding="same", activation="sigmoid")(x)
    decoder = models.Model(decoder_input, decoded_output, name="decoder")

    # Autoencodeur complet
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    autoencoder_output = decoder(encoded)
    autoencoder = models.Model(inputs=autoencoder_input, outputs=autoencoder_output)

     # Compilation du modèle
    autoencoder.compile(optimizer=Nadam(learning_rate=learn_rate), loss=lossfunction, metrics=["accuracy"])
    autoencoder.summary()
    encoder.summary()
    decoder.summary()
    
    #to avoid overtraining
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienceval) 
    # Entraîner le modèle
    history = autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=validation_data_split, shuffle=True,callbacks=[early_stopping])

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
    
    print("Model save.")
else:
    print("Chargement des modèles existants.")
    autoencoder = tf.keras.models.load_model(checkpoint_filepath)


# Afficher des exemples de reconstructions pour évaluer la performance
plot_reconstructions(autoencoder, X_test)

# Test de reconstruction et visualisation des anomalies
erreur = load_and_prepare_image(erreur_dir, image_size)
erreur2 = load_and_prepare_image(erreur_dir2, image_size)
plot_anomaly(autoencoder, erreur)
plot_anomaly(autoencoder, erreur)
