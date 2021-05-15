# -*- coding: utf-8 -*-
"""
Created on Mar 17 2021

Fonctions de traitements classiques (dont ML) de datasets

@author: D0ddos
"""
from time import time

from numpy import shape, reshape, zeros, ones, uint8, int16, float16
from numpy import max as npmax

from scipy.signal import convolve2d
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from munkres import Munkres


def flou(img, matrice=ones((3,3),dtype=float16)/9.0):
    """Floute chaque bande de l'image img[ligne, colonne, bande]"""
    lignes, colonnes, bandes = shape(img)
    
    img_floue = zeros((lignes, colonnes, bandes), dtype=int16)
    
    for bande in range(bandes):
        img_floue[:, :, bande] = convolve2d(img[:, :, bande],
                                            matrice,
                                            'same', 'symm')
    
    return img_floue


def knn(img, X, Y, n_voisins=3):
    lignes, colonnes, bandes = shape(img)
    
    knn = KNeighborsClassifier(n_neighbors=n_voisins)
    t0 = time()
    knn.fit(X, Y)
    print("Temps d'entrainement du knn : {}s".format(time() - t0))
    
    pixels = img.reshape((lignes * colonnes, bandes))
    t0 = time()
    classes = knn.predict(pixels)
    print("Temps d'application du knn : {}s".format(time() - t0))
    
    return classes.reshape((lignes, colonnes))


def kmeans(img, nb_clusters=17):
    """Applique un kmeans à l'image en entrée, retourne un array représentant les clusters."""
    lignes, colonnes, bandes = shape(img)

    X = reshape(img, (lignes * colonnes, bandes))
    
    kmeans = KMeans(nb_clusters, random_state=0)
    kmeans.fit(X)
    
    return reshape(kmeans.predict(X), (lignes, colonnes))


def munkres(array, gt):
    """Applique l'algorithme ongrois pour que les clusters de 'array' corespondent au mieux aux lasses de 'gt'."""
    lignes, colonnes = shape(array)
    nb_clusters = npmax(array) + 1
    
    array_values = reshape(array, (lignes * colonnes))
    gt_values = reshape(gt, (lignes * colonnes))
    
    # Préparation du munkres pour ne pas favoriser de classe
    Mcost =  zeros((nb_clusters, nb_clusters))
    for i in range(lignes*colonnes):
        cluster = array_values[i]
        classe = gt_values[i]
        Mcost[cluster, classe] += 1
    normedMcost = (Mcost.T / Mcost.astype(float16).sum(axis = 1)).T
    
    # Application
    m = Munkres()
    indexes_equivalents = m.compute(1 - normedMcost)
    
    array_out = zeros((lignes, colonnes), dtype=uint8)
    
    # Remplace les clusters par les classes déduites avec Munkres
    for start, end in indexes_equivalents:
        array_out[array == start] = end
    
    return array_out


def maskSansC0(array, gt):
    """Met à zeros tous les pixels de 'array' qui corespondent à la classe 0 sur 'gt'."""
    lignes, colonnes = shape(array)
    array_out = zeros((lignes, colonnes), dtype=uint8)
    
    for l in range(lignes):
        for c in range(colonnes):
            if gt[l, c] != 0:
                array_out[l, c] = array[l, c]
    
    return array_out