# -*- coding: utf-8 -*-
"""
Created on Mar 17 2021

Outils relatifs aux images hyperspectrales et vérités terrains.

@author: D0ddos
"""

from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from numpy import shape, zeros, float32, array
from numpy import max as npmax
from numpy import min as npmin


def openDataset(chemin, nomDictionaire):
    """Récupère le dataset 'nomDictionaire' dans le ficher .mat 'chemin'."""
    return loadmat(chemin)[nomDictionaire]


def saveDataset(dataset, chemin, nomDictionnaire="img"):
    """dataset : tableau numpy, chemin se termine en .mat, """
    savemat(chemin,
            {nomDictionnaire : dataset},
            do_compression=True)
    return None


def createDatasetML(gt, bandes, chemin="centres.txt", voisins=2):
    """Retourne les valeurs de bandes et les classes corespondantes, ce sont des données d'entrainement."""
    X = []
    Y = []
    
    fichierCentre = open(chemin, "r")
    texteCentre = fichierCentre.readlines()[1:]
    fichierCentre.close()
    
    for texte in texteCentre:
        classe, ligne_c, colonne_c = texte.split("\n")[0].split("\t")
        classe, ligne_c, colonne_c = (int(classe), int(ligne_c), int(colonne_c))
        
        # Récupération des données d'entrainement
        for ligne in range(ligne_c - voisins, ligne_c + voisins + 1):
            for colonne in range(colonne_c - voisins, colonne_c + voisins + 1):
                 X.append(bandes[ligne, colonne, :])
                 Y.append(classe)
    return array(X, dtype=float32), array(Y, dtype=float32)


def pca(img_array_np):
    H, W, p = img_array_np.shape
    
    X = img_array_np.reshape((-1, p))
    
    pca = PCA()
    image_pca_sk = pca.fit_transform(X)
    
    return image_pca_sk.reshape((H, W, p))


def normaliserBandes(img):
    """Retourne une verion de img où chaue bande est normalisée."""
    lignes, colonnes, bandes = shape(img)
    img_norm = zeros((lignes, colonnes, bandes), dtype=float32)
    
    for bande in range(bandes):
        mini = npmin(img[:, :, bande])
        img_norm[:, :, bande] = img[:, :, bande] - mini
        maxi = npmax(img_norm[:, :, bande])
        img_norm[:, :, bande] = img_norm[:, :, bande] / maxi
        
    return img_norm