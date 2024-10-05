import torch
import torch.nn as nn
from einops import rearrange, einsum
import ptwt
import pywt
import torch.nn.functional as F


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                 norm='batch', pad_model=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



    

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        w = self.init_w()
        self.dec = nn.Conv2d(1, 4, 2, 2, bias=False)
        self.rec = nn.ConvTranspose2d(4, 1, 2, 2, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = w
                m.weight.requires_grad = False

    def init_w(self):
        ll = torch.ones(1, 1, 2, 2) / 2

        lh = torch.ones(1, 1, 1, 2) / 2
        lh = torch.cat((lh, -lh), 2)

        hl = torch.ones(1, 1, 2, 1) / 2
        hl = torch.cat((hl, -hl), 3)

        hh = torch.ones(1, 1, 1, 1) / 2
        hh = torch.cat((hh, -hh), 2)
        hh = torch.cat((hh, -hh), 3)
        w = torch.cat((ll, lh, hl, hh), 0)
        return w

    def decompose(self, x):
        nb, c = x.size(0), x.size(1)
        x = rearrange(x, 'n (c l) h w-> (n c) l h w', l=1)
        x = self.dec(x)
        x = rearrange(x, '(n c) d h w -> d n c h w', n=nb, c=c)
        x = torch.split(x, 1)
        x = list(map(lambda t: t.squeeze(0), x))
        return x

    def reconstruct(self, coef):
        coef = torch.stack(coef, 2)
        nb, c = coef.size(0), coef.size(1)
        coef = rearrange(coef, 'n c d h w -> (n c) d h w', n=nb, c=c)
        x = self.rec(coef)
        x = rearrange(x, '(n c) l h w-> n (c l) h w', n=nb, c=c)
        return x

def warped_wavedec2(x, wavelet=pywt.Wavelet("haar"), level=1, mode='constant'):
    b, c, h, w = x.shape
    x = x.flatten(0, 1)
    coef = ptwt.wavedec2(x, wavelet, level=level, mode=mode)

    def to_bchw(x):
        return rearrange(x, '(b c) l h w-> l b c h w', b=b, c=c)[0]

    new_coef = [to_bchw(coef[0])]
    for co in coef[1:]:
        new_coef.append(list(map(to_bchw, co)))
    return new_coef


def warped_waverec2(coef, wavelet=pywt.Wavelet("haar")):
    b, c, _, _ = coef[0].shape
    new_coef = [coef[0].flatten(0, 1).unsqueeze(1)]
    for co in coef[1:]:
        new_coef.append(list(map(lambda t: t.flatten(0, 1).unsqueeze(1), co)))
    x = ptwt.waverec2(new_coef, wavelet)
    x = rearrange(x, '(b c) l h w -> l b c h w', b=b, c=c)[0]
    return x


class SDKModule(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.fix_weight = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.dyn_weight = nn.Sequential(
            nn.Conv2d(channels, kernel*kernel, 3, 1, 1),
        )

        self.dyn_tail = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.refine = nn.Conv2d(channels*2, channels, 1)

        # self.gate = nn.Sequential(
        #     nn.Conv2d(channels, channels, 1),
        #     nn.Sigmoid()
        # )

        self.fuse = SSModule(channels)

    def forward(self, coef, ms):

        nb, c, h, w = ms.shape
        fix_ms = self.fix_weight(ms)

        dyn_kernel = self.dyn_weight(ms)
        dyn_kernel = dyn_kernel / dyn_kernel.abs().sum(1, keepdim=True)

        ms = F.unfold(ms, kernel_size=3, padding=1)
        ms = rearrange(ms, 'nb (c k) (h w) -> nb c k h w', c=c, h=h, w=w)
        dyn_ms = einsum(dyn_kernel, ms, 'nb k h w, nb c k h w -> nb c h w')
        dyn_ms = self.dyn_tail(dyn_ms)

        # alpha = self.gate(coef)

        ms = torch.cat((dyn_ms, fix_ms), 1)
        ms = self.refine(ms)
        
        coef = self.fuse(coef, ms)

        return coef


class SSModule(nn.Module):
    def __init__(self, channels, kernel=-1):
        super().__init__()

        # self.margin_prob_head = nn.Conv2d(channels, channels, 1)
        self.joint_prob_head = nn.Conv2d(channels, channels, 3, 1, 1)
        # self.joint_prob_head = nn.Conv2d(channels*2, channels,  1)

    def forward(self, xpan, xms, eps=1e-10):
        # p = torch.cat((xpan, xms), 1)
        p = xpan + xms
        p_joint = self.joint_prob_head(p)
        # p1 = (p_joint / p_lms).sigmoid()
        p2 = (p_joint).sigmoid()
        # xms = p1 * xms + (1-p1) * xpan
        xpan = (1 - p2) * xms + p2 * xpan
        return xpan


class CAModule(nn.Module):
    def __init__(self, dim=5) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(dim*4, dim),
            nn.Sigmoid(),
        )

    def forward(self, all_coef):
        w = self.se(all_coef)
        return w * all_coef


class AFAModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ll_head = SDKModule(channels)
        self.ud_head = SDKModule(channels)
        self.lr_head = SDKModule(channels)
        self.hh_head = SDKModule(channels)
        self.refine = CAModule(4)
        # self.u2 =  nn.Sequential(
        #     # nn.Upsample(scale_factor=(2, 2), mode='bicubic'),
        #     nn.Conv2d(channels, channels*4, 3, 1, 1),
        #     nn.PixelShuffle(upscale_factor=2),
        # )

    def forward(self, coef_list, xms):
        ll, lh, hl, hh = coef_list
        h, w = ll.size(2), ll.size(3)
        ms_reg = xms
        ll = self.ll_head(ll, ms_reg)
        lh = self.ud_head(lh, ms_reg)
        hl = self.lr_head(hl, ms_reg)
        hh = self.hh_head(hh, ms_reg)
        all_coef = [ll, lh, hl, hh]
        all_coef = torch.stack(all_coef, -1)
        all_coef = self.refine(all_coef)
        all_coef = torch.chunk(all_coef, chunks=4, dim=-1)
        ll, lh, hl, hh = map(lambda t: t[..., 0], all_coef)
        new_coef_list = [ll, lh, hl, hh]
        # ms = self.u2(ms)
        return new_coef_list


class Wavelet_Module(nn.Module):
    def __init__(self, channels):
        super(Wavelet_Module, self).__init__()
        self.dwt = DWT()

        subdim = channels // 2
        self.reduce_ms = nn.Sequential(
            nn.AvgPool2d(4, 4),
            nn.Conv2d(channels, subdim, 1),
            nn.ReLU(inplace=True)
            )

        self.reduce_pan = nn.Sequential(
            nn.Conv2d(channels, subdim, 1),
            nn.ReLU(inplace=True)
            )

        self.wave = AFAModule(subdim)
        
        self.wave1 = AFAModule(subdim)
        
        # self.expand_ms = nn.Conv2d(subdim, channels, 1)
        self.expand_pan = nn.Sequential(
            nn.Conv2d(subdim, channels, 1),
            nn.ReLU(inplace=True)
        )
      

    def forward(self, xpan, xms):
        bn, c, h, w = xpan.shape

        xms = self.reduce_ms(xms)
        xpan = self.reduce_pan(xpan)

        coef_l1 = self.dwt.decompose(xpan) # list [4]
        coef_l2 = self.dwt.decompose(coef_l1[0])

        
        coef1 = self.wave(coef_l2, xms)
        recon1 = self.dwt.reconstruct(coef1)


        coef2 = self.wave1(coef_l1, recon1)
        recon2 = self.dwt.reconstruct(coef2)


        recon2 = self.expand_pan(recon2)

        return recon2

# -----------------------------------------------------


class PRNet(nn.Module):
    def __init__(self, spectral_num, channel=32, reg=True):
        super(PRNet, self).__init__()
        print(__file__)

        dims = [channel] * 4

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.PanModule = nn.ModuleList()

        self.MSModule = nn.ModuleList()

        self.FusionModule = nn.ModuleList()

        for i, dim in enumerate(dims):
            if i == 0:
                self.PanModule.append(ConvBlock(1, dim, 3, 1, 1, bias=False))
                self.MSModule.append(ConvBlock(spectral_num, dim, 3, 1, 1, bias=False))
            else:
                self.PanModule.append(ConvBlock(dim, dim, 3, 1, 1, bias=False))
                self.MSModule.append(ConvBlock(dim, dim, 3, 1, 1, bias=False))
            self.FusionModule.append(Wavelet_Module(dim))

        self.out = nn.Conv2d(dims[-1]*2, spectral_num, 1)

        # init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!

    def forward(self, pan, lms, ms):  # x= hp of ms; y = hp of pan

        nb, c, h, w = lms.shape
        xms = lms
        xpan = pan
        
        count = 0
        for pan, ms, wave in zip(self.PanModule, self.MSModule, self.FusionModule):
            xms = ms(xms)
            xpan = pan(xpan)
            xpan = wave(xpan, xms)

        pr = torch.cat((xpan, xms), 1)
        pr = self.out(pr).tanh() + lms

        return pr


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net = PRNet(4).cuda()
    ms = torch.rand(1, 4, 64, 64).cuda()
    pan = torch.rand(1, 1, 256, 256).cuda()
    bms = torch.rand(1, 4, 256, 256).cuda()
    net(pan, bms, ms)
  
    flops = FlopCountAnalysis(net, (pan, bms, ms))

    print(flop_count_table(flops))
