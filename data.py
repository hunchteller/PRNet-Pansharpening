import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import scipy.io as sio

def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        with h5py.File(file_path) as data:  # NxCxHxW = 0x1x2x3
        # tensor type:
            max_val = 1023 if 'gf2' in file_path else 2047
            gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
            gt1 = np.array(gt1, dtype=np.float32) / max_val
            self.gt = torch.from_numpy(gt1)  # NxCxHxW:
            print(self.gt.shape)

            lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
            lms1 = np.array(lms1, dtype=np.float32) / max_val
            self.lms = torch.from_numpy(lms1)

            ms1 = data["ms"][...]  # NxCxHxW
            ms1 = np.array(ms1, dtype=np.float32) /max_val  # NxHxWxC
            self.ms1 = torch.from_numpy(ms1)

            pan1 = data['pan'][...]  # Nx1xHxW
            pan1 = np.array(pan1, dtype=np.float32) / max_val  # Nx1xHxW
            self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
            # import ipdb; ipdb.set_trace()
            print("data range", self.gt.max(), self.gt.min())

            del data, pan1, ms1, lms1, gt1

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index].float(), \
            self.lms[index].float(), \
            self.ms1[index].float(), \
            self.pan[index].float()  # Nx1xHxW:
            # self.pan_hp[index, :, :, :].float(), \
            
        #####必要函数

    def __len__(self):
        return self.gt.shape[0]
    
class Dataset_Pro_Full_Scale(data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        if 'gf2' in file_path:
            scale_factor = 1023
        else:
            scale_factor = 2047
        # tensor type:

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / scale_factor
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1, dtype=np.float32) / scale_factor  # NxHxWxC
        self.ms1 = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / scale_factor  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # print("data range", self.gt.max(), self.gt.min())

    #####必要函数
    def __getitem__(self, index):
        return self.lms[index].float(), \
            self.ms1[index].float(), \
            self.pan[index].float()  # Nx1xHxW:
            # self.pan_hp[index, :, :, :].float(), \
            
        #####必要函数

    def __len__(self):
        return self.pan.shape[0]


if __name__ == '__main__':
    ds = Dataset_Pro('/home/x/Data/Pansharp/data/liang_data/test_data/test_wv3_multiExm1.h5')
    gt, lms, ms, pan = ds[0]
    import ipdb; ipdb.set_trace()
    # from utils.batch_metrics import batch_SSIM, batch_PSNR, _ssim_batch
    # from full_metrics_th import FR_metrics


    # # pan = ds.pan.expand_as(ds.gt)
    # # ssim = _ssim_batch(ds.gt.cuda(), pan.cuda())
    # # print(ssim.shape)
    # #
    # # ssim1 = _ssim_batch((ds.lms.cuda()), pan.cuda())
    # #
    # # diff = ssim - ssim1
    # # print(diff, ssim.mean())
    # import torch.nn.functional as F
    # pr = F.interpolate(ds.ms1, scale_factor=(4, 4), mode='bicubic')
    # psnr = batch_PSNR(pr, ds.gt, 1)
    # psnr1 = batch_PSNR(ds.lms, ds.gt, 1)
    # print(psnr)
    # print(psnr1)