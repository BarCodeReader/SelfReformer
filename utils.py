import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def calculate_mae(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mae = np.mean(np.abs(img1 - img2))

    return mae


class LogWritter:
    def __init__(self, opt):
        self.root = opt.ckpt_root
        os.makedirs(self.root, exist_ok=True)

    def update_txt(self, msg, mode='a'):
        with open('{}/log.txt'.format(self.root), mode) as f:
            f.write(msg)