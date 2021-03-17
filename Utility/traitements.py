# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:28:46 2021

@author: D0ddos
"""

from numpy import shape, zeros, ones, int16, float16
from scipy.signal import convolve2d


def flou(img, matrice=ones((3,3),dtype=float16)/9.0):
    """Floute chaque bande de l'image img[ligne, colonne, bande]"""
    lignes, colonnes, bandes = shape(img)
    
    img_floue = zeros((lignes, colonnes, bandes), dtype=int16)
    
    for bande in range(bandes):
        img_floue[:, :, bande] = convolve2d(img[:, :, bande],
                                            matrice,
                                            'same', 'symm')
    
    return img_floue






