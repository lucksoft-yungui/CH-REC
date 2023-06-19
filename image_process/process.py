import cv2
import numpy as np
import torch
#import torchvision.transforms as transforms

# class ZerosPAD(object):
#     def __init__(self, max_size):
#         self.toTensor = transforms.ToTensor()
#         self.max_size = max_size

#     def __call__(self, img):
#         img = self.toTensor(img)
#         c, h, w = img.shape
#         Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
#         Pad_img[:, :, :w] = img  # right pad

#         return Pad_img
    

mean = 0.588
std = 0.193
inp_w = 320
inp_h = 32

img = cv2.imread("/Users/peiyandong/Documents/code/ai/CRNN_Chinese_Characters_Rec/image_process/213.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = np.array(img)

img = img[:, :, np.newaxis]

height, width = img.shape[:2]
ratio = inp_h/height
new_width = int(width * ratio)
img = cv2.resize(img, (new_width, inp_h), interpolation=cv2.INTER_AREA)


cv2.imshow('img', img)
cv2.waitKey(0)

img = np.reshape(img, (inp_h, new_width, 1))
img = img.astype(np.float32)
img = (img/255. - mean) / std
img = img.transpose([2, 0, 1])


# 调整到0-255的范围并转换为uint8
img_disp = ((img + 1) / 2. * 255.).astype(np.uint8)
img_disp = img_disp.transpose([1, 2, 0])
img_disp = np.squeeze(img_disp)

cv2.imshow('img', img_disp)
cv2.waitKey(0)