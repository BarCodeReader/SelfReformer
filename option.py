import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    # models
    parser.add_argument("--pretrain", type=str, default="") 
    parser.add_argument("--model", type=str, default="network")
    parser.add_argument("--GPU_ID", type=int, default=0)
    parser.add_argument("--pvt_path", type=str, default="./model/pretrain/pvt_v2_b2.pth")

    # dataset
    parser.add_argument("--dataset_root", type=str, default="../dataset/")
    parser.add_argument("--dataset", type=str, default="DUTSTR")
    parser.add_argument("--test_dataset", type=str, default="benchmark_DUTSTE")

    # training setups
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gclip", type=int, default=0)

    # loss
    parser.add_argument("--lmbda", type=int, default=5,
                        help="lambda in loss function, it is divided by 10 to make it float, so here use integer")

    # misc
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--save_every_ckpt", action="store_true") # save ckpt
    parser.add_argument("--save_result", action="store_true") # save pred
    parser.add_argument("--save_all", action="store_true") # save each stage result
    parser.add_argument("--ckpt_root", type=str, default="./ckpt")
    parser.add_argument("--save_root", type=str, default="./output")
    parser.add_argument("--save_msg", type=str, default="")

    return parser.parse_args()


def make_template(opt):

    if opt.random_seed:
        seed = random.randint(0,9999)
        print('random seed:', seed)
        opt.seed = seed

    if not opt.test_only:
        opt.ckpt_root += '/ckpt_rs{}'.format(opt.seed)

    if "network" in opt.model:
        # depth, num_heads, embed_dim, mlp_ratio, num_patches
        opt.transformer = [[2, 1, 512, 3, 49],
                           [2, 1, 320, 3, 196],
                           [2, 1, 128, 3, 784],
                           [2, 1, 64, 3, 3136]]


def get_option():
    opt = parse_args()
    make_template(opt) # some extra configs for the model
    return opt
