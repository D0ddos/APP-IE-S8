# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:02:02 2021

@author: Dorian
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model


def modelRnnSimple(nb_entrees, nb_sorties, nb_neuronesCoucheInterne):
    model = Sequential([
       Dense(nb_neuronesCoucheInterne,
             input_dim=nb_entrees,
             activation='relu'),
       
       Dense(nb_sorties,
             activation='sigmoid')
    ])
    
    return model


def plotModel(model, chemin):
    plot_model(model, to_file=chemin, show_shapes=True)
    return None


