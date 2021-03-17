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

