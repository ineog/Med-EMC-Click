#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:52:35 2025

@author: cota
"""


import cv2
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import label
import pickle
import random

split = "test"
ruta_target = f"/home/cota/datasets/{split}"
archivos = os.listdir(ruta_target)

def generar_tuplas(num_componentes):
    if num_componentes > 1:
        tuplas = [(0, y) for y in range(1, num_componentes+1)]
    else:
        tuplas = [(0,1)]
    return tuplas
#%%

archivos = os.listdir(f"{ruta_target}/gts")
out_path = "/home/cota/datasets/hcuch_dataset_emcclick"
eval_path = "/home/cota/datasets/hcuch_dataset/test_pulmon"
pathh = "/home/cota/datasets/hcuch_dataset/masks_val/masks"
os.makedirs(f'{eval_path}/masks', exist_ok=True)
os.makedirs(f'{eval_path}/images', exist_ok=True)
eval_ = True
# c = 0
xx = 0
n_comp = []
#Extraer imagenes
for ii in range(len(archivos)):
    print(f"\nprocesando {ii+1}/{len(archivos)}")
    archivo = archivos[ii]
    imagen_target = nib.load(f"{ruta_target}/gts/{archivo}")
    datos_target = imagen_target.get_fdata()
    imagen = nib.load(f"/home/cota/datasets/ventaneo/hcuch_{split}/{archivo}")
    datos = imagen.get_fdata()
    z = datos_target.shape[2]
    y = datos_target.shape[1]
    x = datos_target.shape[0]
    # print(f"{archivo[:-7]}")
    ruta_jsons = f"/home/cota/datasets/{split}/labels"
    archivo_json = f"{archivo[:-7]}.json"
    with open(f"{ruta_jsons}/{archivo_json}", 'r') as archivo:
    # Cargar los datos del archivo en un diccionario de Python
        json_ = json.load(archivo)
    n = len(json_)
    for i in json_:
        # print("hola")
        matriz_3d = (datos_target==float(i))
        organo = json_[f'{i}']
        organo__ = "".join((organo.split(",")[1]).split())
      
        if organo__ =='pulmon': 
            ruta_mask = f"{out_path}/{split}/{organo__}/masks"  
            ruta_imagen = f"{out_path}/{split}/{organo__}/images"  
            os.makedirs(ruta_mask, exist_ok=True)
            os.makedirs(ruta_imagen, exist_ok=True)
            for j in range(len(matriz_3d[0,0,:])):
                # print(f"forma plano {matriz_3d[:,:,j].shape}")
                mask = matriz_3d[:,:,j]
                imagen_ = datos[:,:,j]
                # plano = (mask == i)
                if np.any(mask):
                    # print("ajsdas")
                    # cv2.imwrite(f"{ruta_mask}/{xx:05}.jpg", mask.astype(np.uint8))  
                    xx = xx+1#Con componentes conexas
                    # cv2.imwrite(ruta_img, image.astype(np.uint8))
                    # plt.imshow(matriz_3d[:,:,j])
                    # plt.show()
                    # plt.close()
                    etiquetada, num_componentes = label(mask*255)
                    n_comp.append(num_componentes)
                    m_shape = mask.shape
                    etiquetada = etiquetada.reshape((m_shape[0], m_shape[1], 1))
                    if eval_ == False:
                        ret, encoded_image = cv2.imencode('.jpg', etiquetada, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        encoded_image = [encoded_image.reshape(-1,1)]
                        m_shape = etiquetada.shape
                        tuplas = generar_tuplas(num_componentes)
                        if len(tuplas) < 5:
                            num_instances = random.randint(1,len(tuplas))
                        else:
                            num_instances = random.randint(5, len(tuplas))
                        with open(f'{pathh}/{xx:05}.pkl', 'wb') as f:
        # Guardar la imagen (como un array de NumPy) y el array en el archivo
                            pickle.dump((encoded_image, tuplas, num_instances), f)
                        print("Archivo guradado")
                    # break
                    else:
                        n_comp.append(num_componentes)
                        for ii in range(1,num_componentes+1):
                            index = 97+ii-1
                            componente = (etiquetada == ii)
                            if index > 97:
                                letra = chr(index-1)
                                mask = (componente*255).astype(np.uint8)
                                cv2.imwrite(f'{eval_path}/masks/{xx:05}{letra}.jpg', mask)
                                cv2.imwrite(f'{eval_path}/images/{xx:05}{letra}.jpg', imagen_)
                            else:
                                mask = (componente*255).astype(np.uint8)
                                cv2.imwrite(f'{eval_path}/masks/{xx:05}.jpg', mask)
                                cv2.imwrite(f'{eval_path}/images/{xx:05}.jpg', imagen_)
                            
                            
                            
                   
                    # image = (etiquetada*(255/num_componentes)).astype(np.uint8)
                    # print(f"{np.max(image)}")
                    # imagen = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    # cv2.imwrite(f"{ruta_imagen}/{xx:05}.jpg", imagen_)  
                    # plt.imshow(etiquetada)
                    # # plt.title(f"mask_objs: {tuplas}, vu: {np.unique(etiquetada)}, nc: {num_componentes}")
                    # plt.title(f'{xx:05}')
                    # plt.axis('off')
                    # plt.show()
        #ruta_mask = f"{out_path}/{split}/{organo__}/{ii}/masks"         #  Sin componentes conexas
        #ruta_img = f"{out_path}/{split}/{organo__}/{ii}/images"         #
        #os.makedirs(ruta_mask, exist_ok=True)
        #os.makedirs(ruta_img, exist_ok=True)
        # print("jaja",np.shape(matriz_3d))                        #
        #for j in range(z):                                       #
        #    plano = matriz_3d[:,:,j]                             #
        #    if np.any(plano):                                    #
        #        mask = plano*255                                   #
        #        image = datos[:,:,j]                             #
        #        cv2.imwrite(f"{ruta_mask}/{c:05}.jpg", mask.astype(np.uint8))    #
        #        cv2.imwrite(f"{ruta_img}/{c:05}.jpg", image.astype(np.uint8))    #
        #        if organo__ == 'pulmon':
        #            c = c+1

        # cv2.imwrite(ruta_mask, mask.astype(np.uint8))          #Con componentes conexas
        # cv2.imwrite(ruta_img, image.astype(np.uint8))
   
        # dir_mask = f"{out_path}/{split}/masks"
        # dir_image = f"{out_path}/{split}/images"
        # os.makedirs(dir_mask, exist_ok=True)
        # os.makedirs(dir_image, exist_ok=True)
        # # print("organo: ", organo)
        # if organo__ =='pulmon':
        #  for c in range(num_componentes):
        #      componente = (etiquetada == c+1)
        #      # print(c)
        #      #print(f"guardanando componentes conexas:{c+1}/{num_componentes} ")
        #      # Buscar el plano de la componente conexa donde al menos hay un valor True
        #      for z in range(componente.shape[2]):  # Iterar por los planos en el eje Z
        #          if np.any(componente[:, :,z]):  # Si existe al menos un True en el plano
        #              # print(f"\nPlano Z={z} con al menos un valor True:")s
        #              mask = componente[:,:,z]*255
        #              image = datos[:,:,z]
        #              plt.imshow(componente[z,:, : ], cmap='gray', interpolation='none')
        #              # plt.title(f'Plano Z={z} = 3')
        #              # plt.show()
        #              ruta_mask = f"{dir_mask}/{xx:05}.jpg"
        #              ruta_img = f"{dir_image}/{xx:05}.jpg"
        # #             # print(ruta_mask)
        #              cv2.imwrite(ruta_mask, mask.astype(np.uint8))
        #              cv2.imwrite(ruta_img, image.astype(np.uint8))
        #              xx = xx+1