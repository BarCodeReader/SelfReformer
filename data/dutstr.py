import os
import glob
import data


# dataset for DUTS-TR
class DUTSTR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        dir_MASK, dir_IMG = self.get_subdir()
        self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
        self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.jpg")))

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_MASK = "DUTSTR/Masks"
        dir_IMG = "DUTSTR/Images"
        return dir_MASK, dir_IMG