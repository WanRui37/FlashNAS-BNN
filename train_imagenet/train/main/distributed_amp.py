import argparse
import math
import os
import shutil
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from timm.scheduler.cosine_lr import CosineLRScheduler

try:
    from apex import amp, optimizers
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import nets

from utils import KD_loss
from utils.options import parser

# amp
parser.add_argument("--opt-level", type=str)
parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
parser.add_argument("--loss-scale", type=str, default=None)
parser.add_argument("--channels-last", type=bool, default=False)
parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, batch=None, nBatch=None):
    # Warmup
    if epoch < 5:
        lr = args.lr * float(1 + batch + epoch * nBatch) / (5.0 * nBatch)
    else:
        T_total = args.num_epochs * nBatch
        T_cur = (epoch % args.num_epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))  # cosine decay
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
    return lr


def adjust_temperature(model, epoch):
    temperature = torch.ones(1)
    from nets.bnext18 import HardBinaryConv, HardSign  # 根据模型调整

    for module in model.modules():
        if isinstance(module, (HardSign, HardBinaryConv)):
            if (epoch % 1) == 0:
                module.temperature.mul_(0.9)
            temperature = module.temperature
    return temperature


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def main():
    global best_prec1, best_prec5, temperature

    # logger
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p", force=True
    )
    # fh = logging.FileHandler(os.path.join(args.job_dir + 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    print("opt_level = {}\n".format(args.opt_level), flush=True)
    print("keep_batchnorm_fp32 = {}\n".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32), flush=True)
    print("loss_scale = {}\n".format(args.loss_scale), type(args.loss_scale), flush=True)
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()), flush=True)

    # print("args: {}".format(args), flush=True)
    logging.info("args: '{}'".format(args))

    # STAGE_OUT_CHANNEL = os.environ.get('STAGE_OUT_CHANNEL')
    # CONV_TYPE_ARRAY = os.environ.get('CONV_TYPE_ARRAY')
    # STRIDE_ARRAY = os.environ.get('STRIDE_ARRAY')
    # print("STAGE_OUT_CHANNEL: {}".format(STAGE_OUT_CHANNEL), flush=True)
    # print("CONV_TYPE_ARRAY: {}".format(CONV_TYPE_ARRAY), flush=True)
    # print("STRIDE_ARRAY: {}".format(STRIDE_ARRAY), flush=True)

    cudnn.benchmark = True
    best_prec1, best_prec5 = 0, 0
    temperature = 1.0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        # torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    import random

    RANDOM_SEED = 1234  # any random number

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止hash随机化

    set_seed(RANDOM_SEED)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    start_t = time.time()
    # create model
    print("=> creating model '{}'\n".format(args.arch), flush=True)
    model = nets.__dict__[args.arch]()

    if args.arch_teacher is not None:
        print("=> creating teacher model '{}'\n".format(args.arch_teacher), flush=True)
        teacher_model = models.__dict__[args.arch_teacher](pretrained=True)
        for p in teacher_model.parameters():
            p.requires_grad = False
            teacher_model.eval()
        teacher_model = teacher_model.cuda().to(memory_format=memory_format)
    else:
        teacher_model = None

    if args.sync_bn:
        import apex

        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    # args.lr = args.lr*float(args.batch_size*args.world_size)/256.

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or "conv" in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            [{"params": other_parameters, "weight_decay": 0}, {"params": weight_parameters, "weight_decay": 0}],
            lr=args.lr,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [{"params": other_parameters}, {"params": weight_parameters, "weight_decay": args.weight_decay}],
            lr=args.lr,
            betas=(0.9, 0.999),
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        raise ValueError

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / args.epochs), last_epoch=-1)

    """original"""
    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        # for k, v in ckpt['state_dict'].items():
        #     name = 'module.' + k  # add module.
        #     new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    training_loss = []
    training_top1 = []
    training_top5 = []

    testing_loss = []
    testing_top1 = []
    testing_top5 = []

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        if os.path.isfile(args.resume_model):
            print("=> loading checkpoint '{}'".format(args.resume_model))
            checkpoint = torch.load(args.resume_model, map_location="cpu")
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            best_prec5 = checkpoint["best_prec5"]

            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            del new_state_dict
            # optimizer.load_state_dict(checkpoint['optimizer'])

            # training_loss = checkpoint["train_loss"]
            # training_top1 = checkpoint["train_top1"]
            # training_top5 = checkpoint["train_top5"]
            # testing_loss = checkpoint["test_loss"]
            # testing_top1 = checkpoint["test_top1"]
            # testing_top5 = checkpoint["test_top5"]
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_model, checkpoint["epoch"]))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_model))

    # Initialize Amp.
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level=args.opt_level,
        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
        loss_scale=args.loss_scale,
    )

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_kd = KD_loss.DistributionLoss().cuda()
    # adjust the learning rate according to the checkpoint
    for epoch in range(args.start_epoch):
        scheduler.step(epoch=epoch)
    # Data loading code
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")

    crop_size = 224
    val_size = 256

    if args.data_set == "imagenet":
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(), Too slow
                    # normalize,
                ]
            ),
        )
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(val_size),
                    transforms.CenterCrop(crop_size),
                ]
            ),
        )

        train_sampler = None
        val_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        collate_fn = lambda b: fast_collate(b, memory_format)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=collate_fn,
        )
    elif args.data_set == "imagenet_tensor":
        crop_scale = 0.08
        lighting_param = 0.1
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size, scale=(crop_scale, 1.0)),
                    Lighting(lighting_param),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # Too slow
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(val_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        train_sampler = None
        val_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
        )

    if args.evaluate:
        _, prec1, prec5 = validate(val_loader, model, criterion, epoch=-1)
        if args.local_rank == 0:
            print(" * Prec@1 {top1:.3f} Prec@5 {top5:.3f}".format(top1=prec1, top5=prec5))
        quit()

    temperature = 1.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.arch_teacher is None:
            train_obj, train_top1_acc, train_top5_acc = train(
                train_loader, model, criterion, optimizer, epoch, scheduler
            )
        else:
            train_obj, train_top1_acc, train_top5_acc = train_kd(
                train_loader, model, teacher_model, criterion_kd, optimizer, epoch, scheduler
            )
        # evaluate on validation set
        valid_obj, prec1, prec5 = validate(val_loader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            # print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f} Temp {temp}\n'
            #         .format(top1=prec1, top5=prec5, temp=temperature), flush=True)
            logging.info(
                " * Train@1 {train_top1:.3f} Train@5 {train_top5:.3f} Prec@1 {top1:.3f} Prec@5 {top5:.3f} Temp {temp}\n".format(
                    train_top1=train_top1_acc, train_top5=train_top5_acc, top1=prec1, top5=prec5, temp=temperature
                )
            )

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)

            training_loss.append(train_obj)
            training_top1.append(train_top1_acc)
            training_top5.append(train_top5_acc)

            testing_loss.append(valid_obj)
            testing_top1.append(prec1)
            testing_top5.append(prec5)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    # 'train_loss': training_loss,
                    # 'train_top1': training_top1,
                    # 'train_top5': training_top5,
                    # 'test_loss': testing_loss,
                    # 'test_top1': testing_top1,
                    # 'test_top5': testing_top5,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "best_prec5": best_prec5,
                    # 'optimizer' : optimizer.state_dict(),
                },
                is_best,
                args.job_dir,
                epoch=epoch,
            )

            # print("Best Acc:{}%, Temp: {}\n".format(best_prec1, temperature), flush=True)
            logging.info("Best Acc:{}%, Temp: {}\n".format(best_prec1, temperature))
        # 调整模块temperature
        # temperature = adjust_temperature(model, epoch).item()

    training_time = (time.time() - start_t) / 3600
    if not args.distributed or args.local_rank == 0:
        print("total training time = {} hours\n".format(training_time), flush=True)
        print("best acc: {}\n".format(best_prec1), flush=True)


class data_prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    args.print_freq = len(train_loader) // 10  # 总共记录10次
    scheduler.step(epoch=epoch)
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]

    if args.data_set == "imagenet":
        prefetcher = data_prefetcher(train_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1
            # compute output
            output = model(input)
            # print('output:', output)

            # adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']

            loss = criterion(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if i % args.print_freq == 0:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), input.size(0))
                top1.update(to_python_float(prec1), input.size(0))
                top5.update(to_python_float(prec5), input.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    logging.info(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Speed {3:.3f} ({4:.3f})\t"
                        "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            args.world_size * args.batch_size / batch_time.val,
                            args.world_size * args.batch_size / batch_time.avg,
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

            input, target = prefetcher.next()
    else:
        for i, batch in enumerate(train_loader):
            images = batch[0].cuda()
            target = batch[1].cuda()
            # compute output
            output = model(images)

            # adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']

            loss = criterion(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if i % args.print_freq == 0:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), images.size(0))
                top1.update(to_python_float(prec1), images.size(0))
                top5.update(to_python_float(prec5), images.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    logging.info(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Speed {3:.3f} ({4:.3f})\t"
                        "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            args.world_size * args.batch_size / batch_time.val,
                            args.world_size * args.batch_size / batch_time.avg,
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

    return losses.avg, top1.avg, top5.avg


def train_kd(train_loader, model, teacher_model, criterion_kd, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    teacher_model.eval()
    end = time.time()
    args.print_freq = len(train_loader) // 10  # 总共记录10次
    scheduler.step(epoch=epoch)
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]

    if args.data_set == "imagenet":
        prefetcher = data_prefetcher(train_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1
            # compute output
            if args.prof >= 0:
                torch.cuda.nvtx.range_push("forward")
            logits_student = model(input)
            logits_teacher = teacher_model(input)

            # adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']

            loss = criterion_kd(logits_student, logits_teacher)
            # compute gradient and do SGD step
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()  # retain_graph=True

            optimizer.step()

            if i % args.print_freq == 0:
                prec1, prec5 = accuracy(logits_student.data, target, topk=(1, 5))
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), input.size(0))
                top1.update(to_python_float(prec1), input.size(0))
                top5.update(to_python_float(prec5), input.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    logging.info(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Speed {3:.3f} ({4:.3f})\t"
                        "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            args.world_size * args.batch_size / batch_time.val,
                            args.world_size * args.batch_size / batch_time.avg,
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

            input, target = prefetcher.next()
    else:
        for i, batch in enumerate(train_loader):
            images = batch[0].cuda()
            target = batch[1].cuda()
            # compute output
            logits_student = model(images)
            logits_teacher = teacher_model(images)

            # adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']

            loss = criterion_kd(logits_student, logits_teacher)
            # compute gradient and do SGD step
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()  # retain_graph=True

            # clip gradient if necessary
            # parameters_list = []
            # for name, p in model.named_parameters():
            #     if not 'fc' in name:
            #         parameters_list.append(p)
            # adaptive_clip_grad(parameters_list, clip_factor=0.02)

            optimizer.step()

            if i % args.print_freq == 0:
                prec1, prec5 = accuracy(logits_student.data, target, topk=(1, 5))
                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data
                losses.update(to_python_float(reduced_loss), images.size(0))
                top1.update(to_python_float(prec1), images.size(0))
                top5.update(to_python_float(prec5), images.size(0))

                torch.cuda.synchronize()
                batch_time.update((time.time() - end) / args.print_freq)
                end = time.time()

                if args.local_rank == 0:
                    logging.info(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Speed {3:.3f} ({4:.3f})\t"
                        "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                        "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            epoch,
                            i,
                            len(train_loader),
                            args.world_size * args.batch_size / batch_time.val,
                            args.world_size * args.batch_size / batch_time.avg,
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    if args.data_set == "imagenet":
        prefetcher = data_prefetcher(val_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1
            # compute output
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            input, target = prefetcher.next()
    else:
        for i, batch in enumerate(val_loader):
            images = batch[0].cuda()
            target = batch[1].cuda()
            # compute output
            with torch.no_grad():
                output = model(images)
                loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), images.size(0))
            top1.update(to_python_float(prec1), images.size(0))
            top5.update(to_python_float(prec5), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, save, epoch):
    save = os.path.join(save, "checkpoint/")
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpoint.pth.tar")
    torch.save(state, filename)
    # if epoch % 25 == 0:
    #     intermediate_filename = os.path.join(save,'checkpoint_{}.pth.tar'.format(epoch))
    #     shutil.copyfile(filename, intermediate_filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# lighting data augmentation
imagenet_pca = {
    "eigval": np.asarray([0.2175, 0.0188, 0.0045]),
    "eigvec": np.asarray(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting(object):
    def __init__(self, alphastd, eigval=imagenet_pca["eigval"], eigvec=imagenet_pca["eigvec"]):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype("float32")
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), "RGB")
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == "__main__":
    main()
