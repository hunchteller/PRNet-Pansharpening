import os.path as osp

import torch
from torch.utils.data import DataLoader

from data import Dataset_Pro
from utils.batch_metrics import RR_Metrics


class DictMeter():
    def __init__(self, key=None):
        if key is None:
            self.initted = False
        else:
            self.meter = {k: 0 for k in key}
        self.count = 0

    def init_meter(self, kv_dict):
        self.meter = {k: 0 for k in kv_dict.keys()}
        self.initted = True

    def add(self, kv_dict, num):
        if not self.initted:
            self.init_meter(kv_dict)
        for k, v in kv_dict.items():
            self.meter[k] += v * num
        self.count += num

    def avg(self):
        for k, v in self.meter.items():
            self.meter[k] = v / self.count
        return self.meter

    def clear(self):
        self.count = 0
        self.initted = False
        self.meter = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', choices=['qb', 'wv3', 'gf2'])
    args = parser.parse_args()

    dataset = args.ds
    save_dir = f"logs/{args.ds}/prnet"

    ckpt = osp.join(save_dir, 'best_model.pth')

    # ================== Pre-Define =================== #
    batch_size = 32
    device = torch.device('cuda:0')
    spectral_num = 8 if dataset == 'wv3' else 4

    from model.prnet import PRNet

    model = PRNet(spectral_num).to(device)

    model.load_state_dict(torch.load(ckpt))
    print("loading ckpt done")

    meter = DictMeter()

    validate_set = Dataset_Pro(f'/home/x/Data/test_data/test_{dataset}_multiExm1.h5')  # creat data for validation
    # validate_set = Dataset_Pro(f'../../liang_data/validation_data/valid_{dataset}.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=False)  # put training data to DataLoader for batches

    Bit = 10 if dataset == 'gf2' else 11
    maxval = 2 ** Bit - 1
    # maxval = 1
    model.eval()
    with torch.no_grad():
        prlist = []
        for iteration, batch in enumerate(validate_data_loader, 1):
            gt, lms, ms, pan = map(lambda t: t.cuda().float(), batch)

            pr = model(pan, lms, ms)

            pr = (pr * maxval).clip(0, maxval)
            gt = (gt * maxval).clip(0, maxval)
            prlist.append(pr)

            print(pr.max(), pr.min())
            acc_dict = RR_Metrics(pr, gt, Bit)
            meter.add(acc_dict, pr.size(0))

        pr = torch.cat(prlist, 0).detach().cpu()
        torch.save(pr, f"pr_reduced_{args.ds}_{args.model}.pth")

        avg_acc = meter.avg()
        klist = ["Q2N", "SAM", "ERGAS", "SSIM", "PSNR"]
        prt_str = ''
        for k in klist:
            v = avg_acc[k]
            prt_str += f' {v:.4f} &'

        print(prt_str)
