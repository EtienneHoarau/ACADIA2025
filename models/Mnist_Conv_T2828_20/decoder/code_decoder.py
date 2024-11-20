import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#A remplir
model= 'conv_ae_model'
image="4_decompresse.png"

#importation nécessaires
print("chargement du model")
decoder=tf.keras.models.load_model(model + "_decoder")
print("importation de l'image")
encoded_images = np.load('image_compresse.npy')


#décodage de l'image
decoded_images = decoder.predict(encoded_images)

#envoie de l'image dans le fichier decoder
plt.imshow(decoded_images[0], cmap="binary")
