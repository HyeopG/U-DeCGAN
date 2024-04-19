import argparse
from train import *

##
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test", type=str, dest="mode")
parser.add_argument("--train_continue", default=True, type=bool, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=32, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./test", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="denoising", choices=["denoising, DCGAN"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['random', 0.5], dest='opts')
parser.add_argument("--ny", default=128, type=int, dest="ny")
parser.add_argument("--nx", default=128, type=int, dest="nx")
parser.add_argument("--in_channels", default=1, type=int, dest="in_channels")
parser.add_argument("--out_channels", default=1, type=int, dest="out_channels")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt", default=10, type=float, dest="wgt")
parser.add_argument("--norm", default="bnorm", type=str, dest="norm")

parser.add_argument("--network", default="U-DeCGAN", choices=["U-DeCGAN, DCGAN"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)