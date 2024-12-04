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
noise_factor = 0.5

error_transmit =np.load('image_compresse.npy')+noise_factor*np.random.normal(loc=0.0,scale=1.0,size = encoded_images.shape)

#décodage de l'image
decoded_images = decoder.predict(encoded_images)
decoded_error=decoder.predict(error_transmit)
#envoie de l'image dans le fichier decoder
plt.subplot(1,2, 1)
plt.imshow(decoded_images[0], cmap="binary")
plt.subplot(1,2, 2)
plt.imshow(decoded_error[0],cmap="binary")
plt.show()
