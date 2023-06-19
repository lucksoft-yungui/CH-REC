from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
from PIL import Image

class _HWDB(data.Dataset):
   
    def __init__(self, config, phase: str):

        if not (phase in ['train', 'val', 'test']):
            raise AssertionError(phase)
        
        self.data_root_path = config.DATASET.DATA_ROOT_PATH
        self.data_label_path = config.DATASET.DATA_LABEL_PATH
        self.data_file_name = config.DATASET.DATA_FILE_NAME
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.img_c = 1
        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        self.labels = []
        self.phase = phase

        img_id_gt_txt = os.path.join(self.data_label_path, self.data_file_name + "_" + phase + ".txt")

        with open(img_id_gt_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(None, 1)
                img_path = os.path.join(self.data_root_path, line[0])
                if os.path.exists(img_path) and os.stat(img_path).st_size > 0 and line[1]:
                    self.labels.append({img_path: line[1]})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = list(self.labels[idx].keys())[0]
        img = Image.open(img_path)
        img = np.array(img)

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        # if the image has alpha channel, remove it
        if img.shape[2] == 4:
            img = img[:, :, :3]

        height, width = img.shape[:2]
        ratio = self.inp_h/height
        new_width = int(width * ratio)
        img_resize = cv2.resize(img, (new_width, self.inp_h), interpolation=cv2.INTER_AREA)

        # Ensure that grayscale images always have three dimensions
        if img_resize.ndim == 2:
            img_resize = img_resize[:, :, np.newaxis]

        # If we want a grayscale image but the image has three channels, convert it to grayscale
        if self.img_c == 1 and img_resize.shape[2] == 3:
            img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
            img_resize = img_resize[:, :, np.newaxis]

        return img_resize, idx


        # img = np.reshape(img, (self.inp_h, new_width, 1))
        # img = img.astype(np.float32)
        # img = (img/255. - self.mean) / self.std
        # img = img.transpose([2, 0, 1])

        # return img, idx








