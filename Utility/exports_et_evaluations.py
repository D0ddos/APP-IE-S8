# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:28:26 2021

@author: D0ddos
"""

from numpy import zeros, array, shape, reshape, count_nonzero, dot, float32, int8
from numpy import max as npmax
from numpy import min as npmin
from numpy import sum as npsum
from matplotlib.pyplot import imsave, matshow, colorbar, title, show
from matplotlib.pyplot import plot, xlabel, ylabel, fill_between, legend, axis, grid, savefig
from sklearn.metrics import confusion_matrix, accuracy_score
from math import sqrt


def classesToPng(array, nom_fichier="classes.png", couleurs=[]):
    """Sauvegarde une image visible sur le DD représentant les diférentes classes."""
    lignes, colonnes = shape(array)
    
    if couleurs == []:
        couleurs = [[0, 0, 0],
                    [255, 0, 0],#Brocoli 1
                    [255, 94, 0],#Brocoli 2
                    [255, 191, 0],#Fallow
                    [225, 255, 0],#Fallow Rough
                    [128, 255, 0],#Fallow Smooth
                    [34, 255, 0],
                    [0, 255, 64],
                    [0, 255, 157],
                    [0, 255, 255],
                    [0, 162, 255],#Corn
                    [0, 64, 255],#Lettuce 4wk
                    [30, 0, 255],#Lettuce 5wk
                    [128, 0, 255],#Lettuce 6wk
                    [221, 0, 255],#Lettuce 7wk
                    [255, 0, 191],#Vinyard Untrained
                    [255, 0, 98]]#Vinyard Vertical Trellis
        
    img = zeros([lignes, colonnes, 3], dtype="uint8")
    
    for l in range(lignes):
        for c in range(colonnes):
            img[l, c, :] = couleurs[array[l, c]]
    
    imsave(nom_fichier, img)
    return None


def bandesToPng(img, bandes=[0, 1, 2], nom_fichier="export.png"):
    lignes, colonnes, _ = shape(img)
    
    png = zeros([lignes, colonnes, 3], dtype=float32)
    
    for i in [0, 1, 2]:
        mini = npmin(img[:, :, bandes[i]])
        png[:, :, i] = img[:, :, bandes[i]] - mini
        maxi = npmax(png[:, :, i])
        png[:, :, i] = png[:, :, i] / maxi
    
    imsave(nom_fichier, png)
    return None


def confusionMatrix(gt, pred, titre="Confusion matrix", plot=True):
    """gt = gound truth, a 2D array.
    Same for pred = prediction. Return the matrix.
    (cmap cools: hot, gnuplot2)"""
    assert(shape(gt) == shape(pred))
    
    lines, columns = shape(gt)
    gt_1d = reshape(gt, lines * columns)
    pred_1d = reshape(pred, lines * columns)
    
    matrix = confusion_matrix(gt_1d, pred_1d)
    normedMatrix = (matrix.T / matrix.astype(float32).sum(axis=1)).T
    
    if(plot):
        matshow(normedMatrix, cmap="hot")
        colorbar()
        title(titre, y=-0.15)
        show()
    
    return (matrix, normedMatrix)


def accuracySansC0(gt, classesPredites):
    lignes, colonnes = shape(gt)
    
    gt_1d = reshape(gt, lignes*colonnes)
    out_1d = reshape(classesPredites, lignes*colonnes)
    
    gt_1d_s0 = []
    out_1d_s0 = []
    for i in range(lignes*colonnes):
        if(gt_1d[i] != 0):
            gt_1d_s0.append(gt_1d[i])
            out_1d_s0.append(out_1d[i])
    
    return accuracy_score(gt_1d_s0, out_1d_s0)


def spectrographe(img, gt, classe, nom_fichier="Spectre.png", minmax=(0, 8500)):
    """Trace le spectrographe de la classe choisie et l'enregistre"""
    assert(shape(img[:, :, 0]) == shape(gt))
    
    lignes, colonnes, nb_bandes = shape(img)
    MINI, MAXI = minmax
    
    # Création d'un masque pour garder uniquement les pixels de la classe
    masque = zeros((lignes, colonnes), dtype=int8)
    for ligne in range(lignes):
        for colonne in range(colonnes):
            if(gt[ligne, colonne] == classe):
                masque[ligne, colonne] = 1
    
    
    # Calcul du spectre
    n = count_nonzero(masque) # nombre de pixels de cette classe
    luminances = []
    ecartsTypes = []
    for bande in range(nb_bandes):
        img_bande = img[:, :, bande]
        
        img_masquee = img_bande * masque
        
        luminances.append(npsum(img_masquee))
        
        moyenne = luminances[-1] / n
        ecarts = reshape((img_bande - moyenne) * masque, lignes * colonnes)
        sommeCarree = dot(ecarts, ecarts)
        ecartsTypes.append(sqrt(sommeCarree / n))
    
    spectre = (1/n) * array(luminances)
    borne99haute = spectre + 3 * array(ecartsTypes)
    borne99basse = spectre - 3 * array(ecartsTypes)
    
    borne99basse[borne99basse < MINI] = MINI
    borne99haute[borne99haute > MAXI] = MAXI
    
    
    # Dessin
    plot(range(nb_bandes),
         spectre,
         color=[0.4, 0.0, 0.4, 1.0],
         label="Intensité moyenne")
    xlabel("Bande")
    ylabel("Intennsité")
    title("Spectre de la classe {}".format(classe))
    
    # Dispertion des données
    fill_between(range(nb_bandes),
                     borne99haute,
                     borne99basse,
                     color=[1.0, 0.5, 1.0, 0.8],
                     label="Zone des 99%\n(3 fois l'écart-type minoré par 0)")
    
    # Water absorbtion bands
    fill_between([108, 112],
                     MINI, MAXI,
                     color=(.5, .5, .5, .5),
                     label="Bandes d'absorbtion de l'eau\n(non présentes dans la version 'corrected')")
    fill_between([154, 167],
                     MINI, MAXI,
                     color=(.5, .5, .5, .5))
    plot([224, 224],
             [MINI, MAXI],
             color=(.5, .5, .5, .5))
    
    # Mise en forme
    legend(bbox_to_anchor=(0.5, -0.55), loc="lower center")
    axis([0, nb_bandes, MINI, MAXI])
    grid(True,
         'major',
         'both',
         linestyle='dotted')
    
    savefig(nom_fichier,
            format="png",
            dpi=200)
        
    show()
    