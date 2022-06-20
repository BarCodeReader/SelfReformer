import os
import glob
import data


# for vairous benchmark datasets
class Benchmark(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root
        if phase == "test" and opt.test_dataset != "":
            self.name = opt.test_dataset.split('_')[1]
        else:
            self.name = opt.dataset.split('_')[1]

        dir_MASK, dir_IMG = self.get_subdir()
        if self.name == 'HKUIS':
            self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.png")))
        else:
            self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
            self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.jpg")))

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_MASK = "benchmark/{}/Masks".format(self.name)
        dir_IMG = "benchmark/{}/Images".format(self.name)
        return dir_MASK, dir_IMG