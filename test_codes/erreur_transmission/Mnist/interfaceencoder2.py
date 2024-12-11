# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:20:48 2024

@author: Kraline
"""

import sys
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,QHBoxLayout, QLabel, QFileDialog,QMainWindow
from PyQt5.QtGui import QPixmap, QImage,QIcon
from PyQt5.QtCore import Qt

import cv2

# Chemin vers les modèles
encoder_model_path = 'encoder//conv_ae_model_encoder'  # à ajuster avec ton chemin
decoder_model_path = 'decoder//conv_ae_model_decoder'  # à ajuster avec ton chemin

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Charger les modèles d'encodeur et de décodeur
        self.encoder = load_model(encoder_model_path)
        self.decoder = load_model(decoder_model_path)

    def initUI(self):
        self.setWindowTitle('Encodage et Décodage d\'image Mnist')
        self.setGeometry(100, 100, 800, 400)
        # Ajouter l'icône
        self.setWindowIcon(QIcon("carotterape"))
        # Bouton pour sélectionner une image
        self.select_button = QPushButton('Sélectionner une image Mnist', self)
        self.select_button.clicked.connect(self.select_image)
        #Bouton pour sélectionner un fichier compressé
        self.select_compbutton = QPushButton('Sélectionner un fichier déjà encoder', self)
        self.select_compbutton.clicked.connect(self.select_compressedimg)
        # Labels pour afficher les images
        self.original_image_label = QLabel(self)
        self.reconstructed_image_label = QLabel(self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.select_compbutton)
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.reconstructed_image_label)
        layout.addLayout(image_layout)
        self.setLayout(layout)
    def select_compressedimg(self):
        # Ouvrir une boîte de dialogue pour sélectionner une image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image type Mnist", "", "Images (*.npy )", options=options)
        if file_path:
            # Décoder l'image encodée
            encoded_img = np.load(file_path)
            decoded_image = self.decoder.predict(encoded_img)
            decoded_image = np.clip(decoded_image, 0, 1)
            decoded_image = np.squeeze(decoded_image)  # Retirer la dimension inutile
            
            # Afficher l'image reconstruite
            self.original_image_label.clear()
            self.reconstructed_image_label.clear()
            self.display_decoded_image(decoded_image)
            
    def select_image(self):
        # Ouvrir une boîte de dialogue pour sélectionner une image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image Mnist", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            # Charger et afficher l'image originale
            self.display_image(file_path, self.original_image_label)
            
            # Encoder et décoder l'image sélectionnée
            self.process_image(file_path)

    def display_image(self, file_path, label):
        # Charger l'image avec OpenCV et la convertir en QPixmap
        img = cv2.imread(file_path, 0)  # Charger en niveaux de gris
        img = cv2.resize(img, (28, 28))  # Redimensionner pour l'encodeur
        q_image = QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8)
        label.setPixmap(QPixmap.fromImage(q_image).scaled(350, 350, Qt.KeepAspectRatio))
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def process_image(self, file_path):
        # Charger et préparer l'image pour l'encodeur
        img = cv2.imread(file_path, 0)  # Charger en niveaux de gris
        img = cv2.resize(img, (28, 28))  # Redimensionner
        img = np.array(img).astype('float32') / 255.0  # Normaliser
        img = 1 - img  # Inverser pour un fond blanc et chiffre noir
        img = np.expand_dims(img, axis=-1)  # Ajouter la dimension des canaux
        img = np.expand_dims(img, axis=0)   # Ajouter la dimension lot
        
        # Encoder l'image
        encoded_img = self.encoder.predict(img)
        
        # Sauvegarder l'image encodée pour le décodeur
        np.save('decoder/image_compresse.npy', encoded_img)  # Chemin ajusté pour sauvegarder dans le même dossier
        # Décoder l'image encodée
        decoded_image = self.decoder.predict(encoded_img)
        decoded_image = np.clip(decoded_image, 0, 1)
        decoded_image = np.squeeze(decoded_image)  # Retirer la dimension inutile
        
        # Afficher l'image reconstruite
        self.display_decoded_image(decoded_image)

    def display_decoded_image(self, image_data):
        # Convertir l'image reconstruite pour PyQt et l'afficher
        image_data = 1 - image_data 
        q_image = QImage((image_data * 255).astype(np.uint8), image_data.shape[1], image_data.shape[0], QImage.Format_Grayscale8)
        self.reconstructed_image_label.setPixmap(QPixmap.fromImage(q_image).scaled(350, 350, Qt.KeepAspectRatio))
        self.reconstructed_image_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

# Lancer l'application PyQt
app = QApplication(sys.argv)
# Ajouter l'icône
image_app = ImageApp()
image_app.show()
sys.exit(app.exec_())