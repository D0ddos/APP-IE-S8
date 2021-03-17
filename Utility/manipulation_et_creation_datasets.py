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

