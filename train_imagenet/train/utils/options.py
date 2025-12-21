import argparse
import ast
import os

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--gpus",
    type=int,
    nargs="+",
    default=[0],
    help="Select gpu_id to use. default:[0]",
)
parser.add_argument(
    "--data_set",
    type=str,
    default="cifar10",
    help="Select dataset to train. default:cifar10",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/ic619/xk/pytorch/data/cifar-10/",
    help="The dictionary where the input is stored. default:/home/ic619/xk/pytorch/data/cifar-10/",
)
parser.add_argument(
    "--job_dir",
    type=str,
    default="experiments/",
    help="The directory where the summaries will be stored. default:./experiments",
)

# Training
parser.add_argument("--arch", type=str, default="resnet", help="Architecture of model. default:resnet")
parser.add_argument("--arch_teacher", type=str, default=None, help="Architecture of model. default:resnet")
parser.add_argument("--teacher_model", type=str, default=None, help="Path to the model wait for test. default:None")
parser.add_argument("--optimizer", type=str, default="SGD", help="Training optimizer. default:SGD")
parser.add_argument("--epochs", type=int, default=150, help="The num of epochs to train. default:150")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training. default:128")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for MomentumOptimizer. default:0.9")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for train. default:1e-2")
parser.add_argument(
    "--lr_decay_step", type=int, nargs="+", default=[50, 100], help="the iterval of learn rate. default:50, 100"
)
parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay of loss. default:5e-4")
parser.add_argument("--baseline_model", type=str, default=None, help="Path to the model wait for test. default:None")
parser.add_argument(
    "--baseline",
    dest="baseline",
    type=ast.literal_eval,
    help="True or False fine_tune,input should be either True or False",
)
parser.add_argument(
    "--resume",
    dest="resume",
    type=ast.literal_eval,
    help="whether to resume training, input should be either True or False",
)
parser.add_argument("--resume_model", type=str, default=None, help="Path to the model wait for resume. default:None")
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--print-freq", "-p", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--dali", action="store_true", help="use dali as data decoder and argumentation")
parser.add_argument("--dali_cpu", action="store_true", help="dali mode (cpu, gpu)")

# distributed
parser.add_argument(
    "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
)
parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument(
    "--dist-url", default="tcp://224.66.41.62:23456", type=str, help="url used to set up distributed training"
)
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")


# tensorboard
parser.add_argument("--use_tensorboard", dest="use_tensorboard", action="store_true", help="use tensorboard")

# data augument
parser.add_argument("--label-smoothing", default=0.0, type=float, metavar="S", help="label smoothing")
parser.add_argument("--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha")
