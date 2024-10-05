import os
import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F
from data import Dataset_Pro
from utils.logger import Logger
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from model.prnet import PRNet
from utils.batch_metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
from einops import rearrange
from utils.utility import ScalarMeter, DictMeter, print_loss_dict, init_log_as_filename


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    # Parse the arguments

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--ds', choices=['qb', 'gf2', 'wv3'], type=str,
                        help='Path to the config file')
    parser.add_argument('--sleep', default=0, type=int)
    args = parser.parse_args()

    print(args)

    import time

    time.sleep(3600 * args.sleep)

    Bit_Depth = 10 if args.ds == 'gf2' else 11
    dataset = args.ds

    # ds_dir = ds_dict[args.ds]

    torch.backends.cudnn.benchmark = True

    logdir = f'logs/{args.ds}/prnet'

    # Set seeds.
    torch.manual_seed(7)
    np.random.seed(7)
    torch.cuda.manual_seed(7)

    # Setting number of GPUS available for training.
    num_gpus = torch.cuda.device_count()

    # Selecting the model.
    spectral_num = 8 if args.ds == 'wv3' else 4
    model = PRNet(spectral_num)
    # print(f'\n{model}\n')

    # Sending model to GPU  device.
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()

    # Setting up training and testing dataloaderes.
    # print("Training with dataset => {}".format(config["train_dataset"]))

    train_set = Dataset_Pro(f'/home/x/Data/training_data/train_{dataset}.h5')

    train_loader = data.DataLoader(
        train_set,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    test_set = Dataset_Pro(f'/home/x/Data/validation_data/valid_{dataset}.h5')
    # test_set = Dataset_Pro(f'/home/x/Data/Pansharp/data/liang_data/training_data/train_{dataset}.h5')

    test_loader = data.DataLoader(
        test_set,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    # Initialization of hyperparameters.
    start_epoch = 1
    total_iter = 40000
    eval_interval = total_iter // 100

    cur_iter = 0

    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4,
        weight_decay=0,
    )

    # Learning rate sheduler.
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=total_iter // 5,
                                          gamma=0.5)

    criterion = nn.MSELoss()


    # Training epoch.
    def train(cur_iter):
        model.train()
        optimizer.zero_grad()
        meter = ScalarMeter()
        psnr_meter = ScalarMeter()
        while True:
            for i, batch in enumerate(train_loader, 0):
                # Reading data.
                cur_iter = cur_iter + 1
                gt, lms, ms, pan = map(lambda t: t.cuda().float(), batch)

                # gt: [b c h w] lms: [b c h w] ms: [b c h//4 w//4] pan: [b 1 h w]

                pr = model(pan, lms, ms)
                loss = criterion(pr, gt)

                loss = loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if i % 10 == 0:
                    with torch.no_grad():
                        bn = lms.size(0)
                        psnr = batch_PSNR(pr, gt, 1).cpu()
                        meter.add(loss.item(), bn)
                        psnr_meter.add(psnr.mean().item(), bn)

                if cur_iter % eval_interval == 0:
                    writer.add_scalar('Loss/train', meter.avg(), cur_iter)
                    print(
                        f'Iter:{cur_iter}, loss:{meter.avg():.4f}, psnr:{psnr_meter.avg():.4f}, lr:{scheduler.get_last_lr()[0]:.4f}')
                    return cur_iter


    # Testing epoch.

    def test(iteration):
        test_loss = 0.0
        meter_dict = DictMeter()
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader, 0):
                gt, lms, ms, pan = map(lambda t: t.cuda().float(), batch)
                nb, c, h, w = gt.shape
                pr = model(pan, lms, ms)

                loss = criterion(pr, gt)

                acc_dict = RR_Metrics(pr, gt, 1)
                meter_dict.add(acc_dict, gt.size(0))
        # Taking average of performance metrics over test set
        meter = meter_dict.avg()

        test_loss = loss

        # Writing test results to tensorboard
        writer.add_scalar('Loss/test', test_loss, iteration)
        writer.add_scalar('Test_Metrics/CC', meter['CC'], iteration)
        writer.add_scalar('Test_Metrics/SAM', meter['SAM'], iteration)
        writer.add_scalar('Test_Metrics/SSIM', meter['SSIM'], iteration)
        writer.add_scalar('Test_Metrics/ERGAS', meter['ERGAS'], iteration)
        writer.add_scalar('Test_Metrics/PSNR', meter['PSNR'], iteration)
        print(
            f'Test_Iter:{iteration}, loss:{test_loss:.4f}, {print_loss_dict(meter)}')

        # Return Outputs
        metrics = {"loss": float(test_loss),
                   "CC": float(meter['CC']),
                   "SAM": float(meter['SAM']),
                   "ERGAS": float(meter['ERGAS']),
                   "SSIM": float(meter['SSIM']),
                   "PSNR": float(meter['PSNR']),
                   }
        return metrics


    # Setting up tensorboard and copy .json file to save directory.
    PATH = "./" + logdir
    ensure_dir(PATH + "/")
    writer = SummaryWriter(log_dir=PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout

    # Main loop.
    best_psnr = 0.0
    while cur_iter < total_iter:

        cur_iter = train(cur_iter)

        metrics = test(cur_iter)

        torch.save(model.state_dict(), PATH + "/" + "latest.pth")

        # Saving the best model
        if metrics["PSNR"] > best_psnr:
            best_psnr = metrics["PSNR"]

            metrics['best_iter'] = cur_iter
            # Saving best performance metrics
            torch.save(model.state_dict(), PATH + "/" + "best_model.pth")
            with open(PATH + "/" + "best_metrics.json", "w+") as outfile:
                json.dump(metrics, outfile)
