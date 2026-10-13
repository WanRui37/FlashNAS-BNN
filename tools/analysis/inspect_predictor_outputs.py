# Support both module execution and direct script invocation.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os

import torch
from torch.utils.data import DataLoader

from net_predict.net.Predict_RNN import *
from utils.config import Config
from utils.dataset import *


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.1)

    parser.add_argument("--model_arch", type=str, default="OUR", choices=["GCN", "Peephole", "v_SVR", "OUR"])
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet100"]
    )

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])

    parser.add_argument(
        "--out_path", type=str, default="./net_predict_output/"
    )
    parser.add_argument("--data_root", default="./2-npy")
    args = parser.parse_args()

    return args


args = get_parser()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np_best_accuracy_path = os.path.join(args.data_root, args.dataset) + "/" + "2-np_best_accuracy.npy"
np_feature_single_vec_path = os.path.join(args.data_root, args.dataset) + "/" + "2-np_feature_single_vec.npy"
np_feature_total_vec_path = os.path.join(args.data_root, args.dataset) + "/" + "2-np_feature_total_vec.npy"

if __name__ == "__main__":
    model_config = Config()
    batch_size = args.batch_size
    embed_dim = args.embed_dim
    hidden_size = args.hidden_size
    dropout = args.dropout
    args.out_path = os.path.join(args.out_path, args.dataset)

    torch.manual_seed(args.seed)

    if args.model_arch == "GCN":
        model = Predict_GCN(embed_dim, hidden_size, dropout, device).to(device)
    if args.model_arch == "Peephole":
        model = Predict_peephole(embed_dim, hidden_size, dropout, device).to(device)
    elif args.model_arch == "OUR":
        if args.dataset == "imagenet100":
            model = Predict_harsbnn(embed_dim, hidden_size, dropout, device).to(device)
        elif args.dataset in ["cifar10", "cifar100"]:
            model = Predict_harsbnn_cifar(embed_dim, hidden_size, dropout, device).to(device)
    elif args.model_arch == "v_SVR":
        model = Predict_vSVR(embed_dim, hidden_size, dropout, device).to(device)

    state_dict = torch.load(os.path.join(args.out_path, f"CharRNN_{args.model_arch}_best.pth"), map_location=device)

    model.load_state_dict(state_dict)
    model = model.eval()

    train_set, valid_set = get_train_valid_set(
        np_best_accuracy_path, np_feature_single_vec_path, np_feature_total_vec_path, args.dataset, args.ratio
    )
    train_data = DataLoader(train_set, 1, shuffle=False)
    valid_data = DataLoader(valid_set, 1, shuffle=False)

    index = 0
    for data in valid_data:
        x, y = data
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)

        print(f"x.shape: {x.shape}")
        print(f"y.shape: {y.shape}")
        with torch.no_grad():
            output = model(x)
            # for ii in range(output.shape[0]):

            print(
                f"No.{index} predict:{output[0].item()*100:6f} | true:{y.item():6f}"
            )
            # if ii+index==0 or ii+index==1:
            #     print(f"No.{ii+index} {x}{y}")
            # index = index + output.shape[0]
            index = index + 1
