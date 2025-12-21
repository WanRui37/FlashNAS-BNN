import argparse
import logging
import os
import random
import sys

import torch
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from utils.config import Config
from utils.dataset import *
from utils.write2txt import Draw_Linear_Graph, draw_scatter

# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
from net.Predict_RNN import *


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoches", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.8)

    parser.add_argument("--model_arch", type=str, default="OUR", choices=["GCN", "Peephole", "v_SVR", "OUR"])
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "ImageNet16-120", "imagenet100"]
    )

    parser.add_argument("--gpu", type=str, default="5")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    parser.add_argument(
        "--out_path", type=str, default="/mnt/wr/Train_Quant_Imagenet100_184_channel_v8_v1_改1个x/net_predict_output/"
    )
    args = parser.parse_args()

    return args


model_config = Config()
args = get_parser()
learning_rate = args.lr
batch_size = args.batch_size
epoches = args.epoches
embed_dim = args.embed_dim
hidden_size = args.hidden_size
dropout = args.dropout
loss_outpic_path = "./1-loss_lstm.png"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.dataset == "imagenet100":
    layer_len_max = 32
elif args.dataset in ["cifar10", "cifar100"]:
    layer_len_max = 25

output_path = "./net_predict_output/" + args.dataset + "/"
np_best_accuracy_path = "./2-npy/" + args.dataset + "/" + "2-np_best_accuracy.npy"
np_feature_single_vec_path = "./2-npy/" + args.dataset + "/" + "2-np_feature_single_vec.npy"
np_feature_total_vec_path = "./2-npy/" + args.dataset + "/" + "2-np_feature_total_vec.npy"
args.out_path = args.out_path + args.dataset + "/"

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
    print(f"文件夹 {args.out_path} 已成功创建。")
else:
    print(f"文件夹 {args.out_path} 已经存在。")

log_path = os.path.join(args.out_path + "/log.txt")
save_num = 1
if os.path.exists(log_path):
    os.remove(log_path)
else:
    pass

log_format = "%(asctime)s %(message)s"
logging.basicConfig(filename=log_path, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p", force=True)
logging.info("args: '{}'".format(args))
logging.info("CUDA:{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(epoches, model, criterion, scheduler, optimizer, train_data, valid_data, args):

    loss_hist = []
    for epoch in range(epoches):
        model.train()
        train_loss = 0
        batch_count = 0
        scheduler.step(epoch=epoch)

        for param_group in optimizer.param_groups:
            lr = param_group["lr"]

        for data in train_data:
            x, y = data
            x = x.to(device)
            y = y.to(device, dtype=torch.float32)

            output = model(x)

            loss = criterion(output, y.view(-1) / 100)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            print(loss.item(), end="\r")

        train_loss_single = train_loss / batch_size
        logging.info(f"Epoch: {epoch+1}/{epoches}, lr: {lr:.10f} Loss: {train_loss_single:.6f}")
        loss_hist.append(train_loss_single)

        valid_loss_single = valid(epoch, epoches, model, criterion, valid_data, args)

        if epoch % (epoches // save_num) == 0 or epoch == epoches - 1:
            torch.save(model.state_dict(), output_path + "/CharRNN_" + str(args.model_arch) + str(epoch + 1) + ".pth")

        if epoch == 0:
            valid_loss_best = valid_loss_single
            epoch_best = epoch
            torch.save(model.state_dict(), output_path + "/CharRNN_" + str(args.model_arch) + "_best" + ".pth")
        else:
            if valid_loss_single < valid_loss_best:
                torch.save(model.state_dict(), output_path + "/CharRNN_" + str(args.model_arch) + "_best" + ".pth")
                valid_loss_best = valid_loss_single
                epoch_best = epoch
                # print(f"Best loss is epoch{epoch}")
                logging.info(f"Best loss is epoch{epoch}")

    logging.info(f"Best loss:{valid_loss_best:2f} and its epoch{epoch_best}")
    return loss_hist


def valid(epoch, epoches, model, criterion, valid_data, args):
    model.eval()

    valid_loss = 0
    batch_count = 0

    for ii, data in enumerate(valid_data):
        x, y = data
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)

        output = model(x)
        y_single = y.view(-1) / 100
        loss = criterion(output, y_single)
        valid_loss += loss.item()
        if ii == 0:
            output_all = output
            y_all = y_single
        else:
            output_all = torch.cat((output_all, output), dim=0)
            y_all = torch.cat((y_all, y_single), dim=0)

    tau, p_value = kendalltau(output_all.cpu().detach().numpy(), y_all.cpu().detach().numpy())

    r2 = r2_score(output_all.cpu().detach().numpy(), y_all.cpu().detach().numpy())

    if args.mode == "test":
        draw_scatter(
            y_all.cpu().detach().numpy(),
            output_all.cpu().detach().numpy(),
            name=args.out_path + "/scatter_plot_" + str(args.model_arch) + ".pdf",
        )

    valid_loss_single = valid_loss / batch_size
    logging.info(f"Epoch: {epoch+1}/{epoches}, MSE: {valid_loss_single:.6f} tau: {tau:.4f} r2: {r2:.4f}")

    return valid_loss_single


def main():
    set_seed(args.seed)
    print(f"==> mode is [{args.mode}] ...")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"文件夹 {output_path} 已成功创建。")
    else:
        print(f"文件夹 {output_path} 已经存在。")

    # train_set = get_train_set()
    train_set, valid_set = get_train_valid_set(
        np_best_accuracy_path, np_feature_single_vec_path, np_feature_total_vec_path, args.dataset, args.ratio
    )
    # exit()

    train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    valid_data = DataLoader(valid_set, batch_size, shuffle=True, num_workers=4)

    if args.model_arch == "GCN":
        if args.dataset == "imagenet100":
            model = Predict_GCN(embed_dim, hidden_size, dropout, device).to(device)
        elif args.dataset in ["cifar10", "cifar100"]:
            model = Predict_GCN_cifar(embed_dim, hidden_size, dropout, device).to(device)

    if args.model_arch == "Peephole":
        if args.dataset == "imagenet100":
            model = Predict_peephole(embed_dim, hidden_size, dropout, device).to(device)
        elif args.dataset in ["cifar10", "cifar100"]:
            model = Predict_peephole_cifar(embed_dim, hidden_size, dropout, device).to(device)

    elif args.model_arch == "OUR":
        if args.dataset == "imagenet100":
            model = Predict_harsbnn(embed_dim, hidden_size, dropout, device).to(device)
        elif args.dataset in ["cifar10", "cifar100"]:
            model = Predict_harsbnn_cifar(embed_dim, hidden_size, dropout, device).to(device)

    elif args.model_arch == "v_SVR":
        model = Predict_vSVR(embed_dim, hidden_size, dropout, device).to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/epoches), last_epoch=-1)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoches, warmup_t=64, warmup_lr_init=1e-3)
    if args.mode == "train":
        loss_hist = train(epoches, model, criterion, scheduler, optimizer, train_data, valid_data, args)
        Draw_Linear_Graph([loss_hist, "train_loss"], output_file_path=loss_outpic_path)
    elif args.mode == "test":
        state_dict = torch.load(args.out_path + "/CharRNN_" + str(args.model_arch) + "_best" + ".pth")
        # state_dict = torch.load(args.out_path + "/CharRNN_" + str(args.model_arch) + "256" + ".pth")

        model.load_state_dict(state_dict)

        valid(0, epoches, model, criterion, valid_data, args)


if __name__ == "__main__":
    main()
