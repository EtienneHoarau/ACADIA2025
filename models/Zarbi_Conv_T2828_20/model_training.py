import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Charger les données MNIST
X_train, Y_train = np.load("./zarbi/X_train.npy"), np.load("./zarbi/Y_train.npy")
X_valid, Y_valid = np.load("./zarbi/X_test.npy"), np.load("./zarbi/Y_test.npy")
X_train,Y_train = X_train.astype('float32') / 255.0,Y_train.astype('float32') / 255.0
X_valid,Y_valid = X_valid.astype('float32') / 255.0,Y_valid.astype('float32') / 255.0

# Chemin de fichier de sauvegarde formaliser
checkpoint_filepath = 'conv_ae_model'

#Création model

#vérifie existance du modele 
if not os.path.exists(checkpoint_filepath):
    print("Pas de fichier de sauvegarde\n")
    
    # Définir l'autoencodeur
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=(28, 28, 1))  # Reshape pour MNIST (grayscale, donc 1 canal)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(encoder_input)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # sortie : 14 x 14 x 16
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # sortie : 7 x 7 x 32
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # sortie : 3 x 3 x 64
    x = tf.keras.layers.Conv2D(30, 3, padding="same", activation="relu")(x)
    encoded_output = tf.keras.layers.GlobalAvgPool2D()(x)  # sortie : 30
    
    # Définir l'encodeur (model partiel)
    conv_encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded_output, name="encoder")
    
    # Décodeur
    decoder_input = tf.keras.layers.Input(shape=(30,))
    x = tf.keras.layers.Dense(3 * 3 * 16)(decoder_input)
    x = tf.keras.layers.Reshape((3, 3, 16))(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    decoded_output = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same")(x)
    
    # Définir le décodeur (model partiel)
    conv_decoder = tf.keras.Model(inputs=decoder_input, outputs=decoded_output, name="decoder")

    # Autoencodeur complet
    autoencoder_input = encoder_input
    autoencoder_output = conv_decoder(conv_encoder(autoencoder_input))
    conv_ae = tf.keras.Model(inputs=autoencoder_input, outputs=autoencoder_output)

    # Compiler le modèle
    conv_ae.compile(loss="mse", optimizer="nadam", metrics="accuracy")
    
    # Entraîner le modèle
    conv_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))

    #sauvegarde le model total au cazu
    conv_ae.save(checkpoint_filepath, save_format='tf')
    #envoie le modele de l'encoder dans le fichier encoder
    conv_encoder.save("./encoder/"+checkpoint_filepath + "_encoder", save_format='tf')
    #envoie le modele du decoder dans le fichier decoder
    conv_decoder.save("./decoder/"+checkpoint_filepath + "_decoder", save_format='tf')
    print("Fichiers sauvegardés")
else:
    print("Fichier modèle existant")
    #charge l'AE si il existe dans le fichier global
    conv_ae = tf.keras.models.load_model(checkpoint_filepath)
    #charge l'encodeur si il existe dans son fichier respectif
    conv_encoder = tf.keras.models.load_model("./encoder/"+checkpoint_filepath + "_encoder")
    #charge le decodeur si il existe dans son fichier respectif
    conv_decoder = tf.keras.models.load_model("./decoder/"+checkpoint_filepath + "_decoder")

# Fonction pour visualiser les reconstructions pour test model
def plot_reconstructions(model, images, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    print(reconstructions.shape)
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

#Résume l'AE pour voir les différentes entrées sorties
conv_ae.summary()
#Affiche les images de test avec leurs versions décompréssés pour test
plot_reconstructions(conv_ae, X_valid)
plt.show()
