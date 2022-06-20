import random
import numpy as np
import cv2
import torch
from torchvision import transforms


class Augment:
    def __init__(self, opt):
        self.opt = opt
        self.psize = self.opt.img_size
        self.size = (self.psize, self.psize)
        self.norm = transforms.Normalize(mean=(0.485, 0.458, 0.407),
                                         std=(0.229, 0.224, 0.225))

    def norm(self, x):
        return self.norm(x)

    def to_tensor(self, x):
        # x is np array
        tensor = torch.from_numpy(x).float()
        return tensor

    def resize(self, IMG):
        resize_IMG = cv2.resize(IMG, (self.psize, self.psize), interpolation=cv2.INTER_LINEAR)

        return resize_IMG.copy()

    def crop(self, MASK, IMG):

        dice = random.random()
        h, w = IMG.shape[:-1]

        if dice < .1 and w > self.psize and h > self.psize:
            new_h = random.randrange(0, h - self.psize)
            new_w = random.randrange(0, w - self.psize)

            crop_MASK = MASK[new_h:new_h + self.psize, new_w:new_w + self.psize]
            crop_IMG = IMG[new_h:new_h + self.psize, new_w:new_w + self.psize, :]
        else:
            crop_IMG = cv2.resize(IMG, (self.psize, self.psize), interpolation=cv2.INTER_LINEAR)
            crop_MASK = cv2.resize(MASK, (self.psize, self.psize), interpolation=cv2.INTER_LINEAR)
            crop_MASK = (np.array(crop_MASK) > 127).astype(np.float64)

        return crop_MASK.copy(), crop_IMG.copy()

    def gen_ctr(self, IMG, kernel_size=5):
        # gen contour for numpy based array
        kernel = np.ones((kernel_size, kernel_size))
        C = cv2.dilate(IMG, kernel) - cv2.erode(IMG, kernel)

        return C

    def flip_and_rotate(self, MASK, IMG):
        # h,w,c
        hflip = random.random() < 0.5
        # vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            MASK, IMG = MASK[:, ::-1], IMG[:, ::-1, :]
        # if vflip:
        #    MASK, IMG = MASK[::-1, :], IMG[::-1, :, :]
        if rot90:
            MASK, IMG = MASK.transpose(1, 0), IMG.transpose(1, 0, 2)

        return MASK.copy(), IMG.copy()