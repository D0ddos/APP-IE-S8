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


def effetPcaKnnKmeans():
    import matplotlib.pyplot as plt
    
    img = datasets.openDataset("../Datasets/Salinas.mat", "salinas")
    gt = datasets.openDataset("../Datasets/Salinas_gt.mat", "salinas_gt")
    
    nb_comp_a_tester = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100, 200]
    acc_3nn = []
    acc_kmeans = []
    
    img_pca = datasets.pca(img)
    
    for nb_composantes in nb_comp_a_tester:
        print(nb_composantes)
        img_temp_pca = img_pca[:, :, :nb_composantes]
        
        X, Y = datasets.createDatasetML(gt, img_temp_pca, "../Datasets/centres.txt")
        
        img_temp_3nn = traitements.knn(img_temp_pca, X, Y)
        img_temp_kmeans = traitements.munkres(traitements.kmeans(img_temp_pca), gt)
        
        acc_3nn.append(ioEval.accuracySansC0(gt, img_temp_3nn))
        acc_kmeans.append(ioEval.accuracySansC0(gt, img_temp_kmeans))
    
    plt.plot(nb_comp_a_tester, acc_3nn, "--r+", label="3nn")
    plt.plot(nb_comp_a_tester, acc_kmeans, "--b+", label="kmeans")
    plt.legend()
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Accuracy (entre 0 et 1)")
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((0,200,0,1))
    plt.grid("..")
    plt.show()


def showPCA():
	img = datasets.openDataset("../Datasets/Salinas.mat", "salinas")
	img_pca = datasets.pca(img)
	ioEval.bandesToPng(img_pca, [0, 1, 2], "../Workspace/pca.png")