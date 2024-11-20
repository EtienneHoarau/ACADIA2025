import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#A remplir
model= 'conv_aeDebruiteur_model'
image="1-2.png"

#importation nécessaires
print("chargement du model")
encoder=tf.keras.models.load_model(model)
print("importation de l'image")
img = cv2.imread(image,0)

#traitement de l'image
img = cv2.resize(img,(28,28))
plt.imshow(img,cmap="binary")
img = np.array(img).astype('float32')/255.0
img = 1 - img
#img = np.expand_dims(img, axis=-1)
#img = np.expand_dims(img, axis=0)
#vérification de l'image
plt.figure()
plt.subplot(2, 1,1)
plt.imshow(np.reshape(img, (28, 28)),cmap='binary')
#encodage de l'image
print(f"shape image= {img.shape}")
encoded_img = encoder.predict(img)
#envoie de l'image dans le fichier decoder
#np.save('../decoder/image_compresse.npy', encoded_img)
plt.subplot(2,1, 2)
plt.imshow(encoded_img, cmap="binary")
print("Image envoyé")