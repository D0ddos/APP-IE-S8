# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:22:15 2021

@author: vexco
"""

def PCA(img_array_np):

    
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

