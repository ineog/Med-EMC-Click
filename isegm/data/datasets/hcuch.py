# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:21:59 2025

@author: misep
"""

from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
import os
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class HcuchDataset(ISDataset):
    def __init__(self, dataset_path, split='masks_train',img_split='img_train',stuff_prob=0.0,
                 allow_list_name=None, anno_file='hannotation.pickle', **kwargs):
        super(HcuchDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.img_split_path = dataset_path / img_split
        self.split = split
        self.img_split = img_split
        #self._images_path = self._split_path / 'images'
        self._images_path = self.img_split_path / 'images'

        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob
        #print(self._images_path)
        self.dataset_samples = sorted(os.listdir(self._images_path))

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                    if sample[0] in allow_images_ids]

    def get_sample(self, index) -> DSample:

        image_id = self.dataset_samples[index][:-4]
        #print("IMAGE ID", image_id)
        image_path = self._images_path / f'{image_id}.jpg'

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        packed_masks_path = self._masks_path / f'{image_id}.pkl'
        #print("image id:",image_id)

        # with open(packed_masks_path, 'rb') as f:
        with open(packed_masks_path, 'rb') as f:
    # Cargar la imagen (como un array de NumPy) y el array
            encoded_layers, objs_mapping, num_instances = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)
        #print("primerp",layers.shape)
        #print(layers.shape)
        #print("max",np.max(layers))
        #print("packed_masks_path:",packed_masks_path)
        #print("size1:", layers)
        #print("size2:", np.size(layers))
        #layers = np.stack(layers, axis=2)
        #print(layers.shape)
        #objs_mapping = [(0, 1)]
        instances_info = {0:None}

        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]
        #print(f"num_instances:{num_instances}")
        if self.stuff_prob > 0 and random.random() < self.stuff_prob: #whether sample some non-object
            for inst_id in range(num_instances, len(objs_mapping)):
                instances_info[inst_id] = {
                    'mapping': objs_mapping[inst_id],
                    'parent': None,
                    'children': []
                }
        else:
            for inst_id in range(1, len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        #return DSample(image, layers, objects_ids=np.unique(layers), sample_id=index)
        return DSample(image, layers, objects = instances_info)
