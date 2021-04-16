# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:27:54 2021

@author: D0ddos
"""

import manipulation_et_creation_datasets as datasets
import traitements
import exports_et_evaluations as ioEval


def creationDatasetFlou3x3():
    img = datasets.openDataset("../Datasets/Salinas.mat", "salinas")
    img_floue = traitements.flou(img)
    datasets.saveDataset(img_floue, "../Datasets/Flous/Salinas_flou_3x3.mat")
    return None


def exportClassesGt():
    img = datasets.openDataset("../Datasets/Salinas_gt.mat", "salinas_gt")
    ioEval.classesToPng(img, "../Workspace/classes_gt.png")
    return None


def kmeansMunkresPropre():
    img = datasets.openDataset("../Datasets/Flous/Salinas_flou_3x3.mat", "img")
    gt = datasets.openDataset("../Datasets/Salinas_gt.mat", "salinas_gt")
    
    clusters = traitements.kmeans(img)
    bonnes_classes = traitements.munkres(clusters, gt)
    
    propre = traitements.maskSansC0(bonnes_classes, gt)
    
    ioEval.classesToPng(propre, "../Workspace/kmeans_3x3_munkres_propre.png")
    ioEval.confusionMatrix(gt, bonnes_classes, "Matrice de confusion kmeans")
    return None


def knnPropre():
    img = datasets.openDataset("../Datasets/Salinas.mat", "salinas")
    gt = datasets.openDataset("../Datasets/Salinas_gt.mat", "salinas_gt")
    
    X, Y = datasets.createDatasetML(gt, img, "../Datasets/centres.txt")
    knnResult = traitements.knn(img, X, Y)
    
    propre = traitements.maskSansC0(knnResult, gt)
    
    ioEval.classesToPng(propre, "../Workspace/3nn_propre.png")
    ioEval.confusionMatrix(gt, knnResult, "matrice de confusion 3nn")
    return None

def PCA_array(img_array_np):
        
    from sklearn.decomposition import PCA
    
    H, W, p = img_array_np.shape
    
    # aplatir
    X = img_array_np.reshape((-1, p))
    
    # centrage + PCA en meme temps
    pca = PCA()
    image_pca_sk = pca.fit_transform(X)
    
    # je remets l'image à sa taille d'origine
    image_pca_sk = image_pca_sk.reshape((H, W, p))
    
    return image_pca_sk[:, :, :]
