# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:29:02 2021

@author: D0ddos
"""

from scipy.io import loadmat, savemat


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
    return X, Y

