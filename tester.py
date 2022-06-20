import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_loader
from tqdm import tqdm
from utils import calculate_mae


class Tester():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)

        self.test_loader = generate_loader("test", opt)

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt

        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.save_msg)
            os.makedirs(save_root, exist_ok=True)

        mae = 0
        
        for i, inputs in enumerate(tqdm(self.test_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]

            b, c, h, w = MASK.shape
            
            pred = self.net(IMG)
            
            mask = (MASK*255.).squeeze().detach().cpu().numpy().astype('uint8')
            pred_sal = F.pixel_shuffle(pred[-1], 4)
            pred_sal = F.interpolate(pred_sal, (h,w), mode='bilinear', align_corners=False)
            pred_sal = torch.sigmoid(pred_sal).squeeze()
            pred_sal = (pred_sal * 255.).detach().cpu().numpy().astype('uint8')
            
            matt_img = pred[0].repeat(1,256,1,1)
            matt_img = F.pixel_shuffle(matt_img, 16)
            matt_img = F.interpolate(matt_img, (h,w), mode='bilinear', align_corners=False)
            matt_img = torch.sigmoid(matt_img)
            matt_img = (matt_img*255.).squeeze().detach().cpu().numpy().astype('uint8')

            if opt.save_result:
                save_path_msk = os.path.join(save_root, "{}_msk.png".format(NAME))
                save_path_matt = os.path.join(save_root, "{}_matt.png".format(NAME))
                io.imsave(save_path_msk, mask)
                io.imsave(save_path_matt, matt_img)
                
                if opt.save_all:
                    for idx, sal in enumerate(pred[1:]):
                        scale=224//(sal.shape[-1])
                        sal_img = F.pixel_shuffle(sal,scale)
                        sal_img = F.interpolate(sal_img, (h,w), mode='bilinear', align_corners=False)
                        sal_img = torch.sigmoid(sal_img)
                        sal_path = os.path.join(save_root, "{}_sal_{}.png".format(NAME, idx))
                        sal_img = sal_img.squeeze().detach().cpu().numpy()
                        sal_img = (sal_img * 255).astype('uint8')
                        io.imsave(sal_path, sal_img)
                else:
                    # save pred image
                    save_path_sal = os.path.join(save_root, "{}_sal.png".format(NAME))
                    io.imsave(save_path_sal, pred_sal)
            mae += calculate_mae(mask, pred_sal)
            
        return mae/(len(self.test_loader)*255.)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return

