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

# Paramètres de base
image_size = (256,256)
data_dir = "D:\\cours\\PLP\\dataset\\hazelnut\\train\\augmented_images"  # Dossier de visages
checkpoint_filepath = 'autoencoder_faces_512_18'
encoder_filepath = "./encoder/" + checkpoint_filepath + "_encoder"
decoder_filepath = "./decoder/" + checkpoint_filepath + "_decoder"

IMG_HEIGHT, IMG_WIDTH = image_size
BATCH_SIZE = 1
EPOCHS = 200
NB_IMAGE = 3000

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

# Chargement et prétraitement des images
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
    if color_mode == 'rgb':
        return images.reshape(-1, image_size[0], image_size[1], 3)  # Reshape pour Conv2D (RGB)
    else:
        return images.reshape(-1, image_size[0], image_size[1], 1)  # Reshape pour Conv2D (Grayscale)

# Charger les images et les diviser en ensembles d'entraînement et de test
X_train2 = load_images(data_dir, image_size, color_mode='rgb')
plt.figure();

X_test = X_train2[:int(NB_IMAGE/2)]
X_train = X_train2[int(NB_IMAGE/2):]
#plt.imshow(X_test[1])
plt.imshow(X_train[1])
plt.figure();
print("Printed")

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
    autoencoder.summary()
    encoder.summary()
    decoder.summary()
    # Entraîner le modèle
    history = autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

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
plot_reconstructions(autoencoder, X_train)

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

# Test de reconstruction et visualisation des anomalies
erreur_dir = "D:\\cours\\PLP\\dataset\\hazelnut\\test\\cut\\006.png"
erreur = load_and_prepare_image(erreur_dir, image_size)
[recon,diff_image]=plot_anomaly(autoencoder, erreur)


save_exemple="oui"
if(save_exemple=="oui"):
    
    comp_test=X_train[0]
    comp_test = np.expand_dims(comp_test, axis=0)
    diff_image = np.expand_dims(diff_image, axis=-1)    
    decomp_test=autoencoder.predict(comp_test)
    

    psnr_comp = calculate_psnr(comp_test, decomp_test)

    # Convert outputs from 0-1 to 0, 255
    recon *= 255.0
    diff_image *= 255.0
    comp_test *= 255.0
    decomp_test *= 255.0
    erreur *= 255.0

    # Convert floats to bytes
    # recon = recon.astype(np.uint8)
    # diff_image = diff_image.astype(np.uint8)
    # comp_test = comp_test.astype(np.uint8)
    # decomp_test = decomp_test.astype(np.uint8)
    # erreur = erreur.astype(np.uint8)



    # Transpose the images from channel first (4, 96, 96) to channel last (96, 96, 4)
    image1 = recon[0, :, :, :]
    image2 = diff_image[0, :, :, :]
    image3 = comp_test[0, :, :, :]
    image4 = decomp_test[0, :, :, :]
    image5 = erreur[0, :, :, :]
    
    path_save_img = "D:\\cours\\PLP\\pour_coudoux" 

    cv2.imwrite("D:\\cours\\PLP\\pour_coudoux\image_a_detecter.png", image5)    
    cv2.imwrite('D:\\cours\\PLP\\pour_coudoux\image_reconstruite.png', image1)
    cv2.imwrite('D:\\cours\\PLP\\pour_coudoux\difference.png', image2)
    cv2.imwrite('D:\\cours\\PLP\\pour_coudoux\image_a_compresser.png', image3)
    cv2.imwrite('D:\\cours\\PLP\\pour_coudoux\image_decompresser.png', image4) 
    file = open("D:\\cours\\PLP\\pour_coudoux\psnr.txt", "w")  
    file.write(str(psnr_comp))
    file.close()
    
