# -*- coding: utf-8 -*-
"""
Created on Apr 16 2021

Fonctions utiles aux réseaux de neurones

@author: Dorian
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from numpy import array, float32


def modelRnnSimple(nb_entrees, nb_sorties, nb_neuronesCoucheInterne):
    """Crée un modèle simple avec une couche cachée de ReLU."""
    model = Sequential([
       Dense(nb_neuronesCoucheInterne,
             input_dim=nb_entrees,
             activation='relu'),
       
       Dense(nb_sorties,
             activation='sigmoid')
    ])
    
    return model


def plotModel(model, chemin):
    """Représente le modèle (pas testé)"""
    plot_model(model, to_file=chemin, show_shapes=True)
    return None


def classeToActivatedVector(Y, nb_classes=16, classe_min=1):
    """Y est une liste de classes (pour entrainer un modèle de ML).
    Retourne une liste contenant des vecteurs qui représentent les classes."""
    Y_vect = []
    
    for y in Y:
        Y_vect.append([])
        for i in range(nb_classes):
            Y_vect[-1].append(0.0)
        Y_vect[-1][int(y) - classe_min] = 1.0
    
    return array(Y_vect, dtype=float32)