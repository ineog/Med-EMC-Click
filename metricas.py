#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:34:27 2024

@author: cota
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import pickle
#%%
def TP_FP_TN_FN(mask, truth):
    mask_binary = mask.astype(bool)
    ground_truth_binary = truth.astype(bool)
    TP = np.sum((mask_binary == 1) & (ground_truth_binary == 1))
    FP = np.sum((mask_binary == 1) & (ground_truth_binary == 0))
    TN = np.sum((mask_binary == 0) & (ground_truth_binary == 0))
    FN = np.sum((mask_binary == 0) & (ground_truth_binary == 1))
    
    return TP, FP, TN, FN

#%%
u = 150
ckpt = '002_ckpt'
ruta_pred = f'/home/cota/EMC-Click/experiments/evaluation_logs/others/{ckpt}/predictions_vis/test_pulmon/matrices'
ruta_truth = '/home/cota/datasets/hcuch_dataset/test_pulmon/masks'
organo = "pulmon"
# def calculate_metrics(ruta_pred, ruta_truth, organo):   
archivos_pred = sorted((os.listdir(ruta_pred)), reverse=True)
archivos_truth = sorted(os.listdir(ruta_truth))
for i in range(len(archivos_truth)):
    archivos_truth[i] = (f'{i:04}.jpg', archivos_truth[i])
names = []
c = 0
for file_truth, _ in archivos_truth:
    id_t, _ = file_truth.split(".")
    y = next((s for s in archivos_pred if id_t in s), None)
    names.append(y)
    c +=1
    # print(c,"/5094")

dice = []
beetwen = []
P = []
# S = []
R = []
A = []
AP = []
ID = []
 # print("Dice:", coeficiente_dice(mask, kk[:,:,2]))
for i in range(len(names)):
    # mask_pred = np.loadtxt(f"{ruta_pred}/{names[i]}") > 0.5
    with open(f"{ruta_pred}/{names[i]}", 'rb') as f:
        encoded_layers = pickle.load(f)

    nparr = np.frombuffer(encoded_layers, np.uint8)
    mask_pred = ((cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE))/255) >0.5
    mask_truth = cv2.imread(f"{ruta_truth}/{archivos_truth[i][1]}")[:,:,0]
    # print(f"Metricas entre {names[i]} y {archivos_truth[i][0]}")
    beetwen.append(f"Metricas entre {names[i]} y {archivos_truth[i][1]}")
     
    TP, FP, TN, FN = TP_FP_TN_FN(mask_pred, mask_truth)
    area = TP + FN
    dc = 2*TP/(2*TP+FP+FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # specifity = TN /(FP + TN) if(FP + TN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    areapred = TP + FP
    P.append(precision)
    dice.append(dc)
     # S.append(specifity)
    R.append(recall)
    A.append(area)
    AP.append(areapred)
    ID.append(names[i])
data = {'Dice coeficient': dice,
         'Precision': P,
         'Recall': R,
         'Area': A,
         'Areapred': AP,
         'ImageID':ID,        
         }
data = pd.DataFrame(data)
# organ = organo
#%%
df__= data[data['Areapred'] >= u]
df__= df__[df__['Area'] >= u]
mean_precision = np.mean(df__['Precision'])
mean_recall = np.mean(df__ ['Recall'])
mean_dice =np.mean( df__['Dice coeficient'])
print(f"Metricas {ckpt}\n\nPrecision:{mean_precision}\nRecall:{mean_recall}\nDice:{mean_dice}")
#%%
plt.scatter(data['Areapred'], data['Recall'])
plt.title(f'scatterplot gamma 2')
plt.xlabel('Area pred')
plt.ylabel('Recall')
plt.show()
plt.close()
plt.scatter(data['Areapred'], data['Precision'])
plt.title(f'scatterplot gamma 2')
plt.xlabel('Area pred')
plt.ylabel('Precision')
plt.show()
