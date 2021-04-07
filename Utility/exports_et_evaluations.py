# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:28:26 2021

@author: D0ddos
"""

from numpy import zeros, shape, reshape, float32
from matplotlib.pyplot import imsave, matshow, colorbar, title, show
from sklearn.metrics import confusion_matrix


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
