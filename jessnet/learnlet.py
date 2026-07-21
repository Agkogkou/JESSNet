"""Spherical-wavelet learnlet network.

A learned, wavelet-like denoiser. The input is a multiscale patch cube
(batch, n_scales, H, W) obtained from the spherical wavelet transform of a
map (see harmonics.wt_trans + patches). The network learns per-scale analysis
and synthesis filters and per-scale threshold coefficients, and returns a
denoised map patch (batch, 1, H, W).

The `sigma` argument is the noise level; the hard-threshold level at scale q is
`k_q * sigma`, where the `k_q` are learned by the small MiniNet.
"""

import os

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class MiniNet(nn.Module):
    """Learns the per-scale threshold multipliers k_q from a constant input."""

    def __init__(self, n=4):
        super(MiniNet, self).__init__()
        self.dense = nn.Linear(1, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, n)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.last_act(x)
        return x


class NormalizeByL2(nn.Module):
    def forward(self, weight):
        return weight / (weight.norm(dim=(2, 3), keepdim=True) + 1e-8)


class Learnlet(nn.Module):

    def __init__(self, n_scales=5, kernel_size=5, filters=64, exact_rec=True,
                 thresh='hard', pretrained=True, device='cpu'):
        super(Learnlet, self).__init__()

        self.exact_rec = exact_rec
        self.thresh = thresh
        starlet = torch.FloatTensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])[None, :]
        starlet2d = torch.matmul(starlet.T, starlet)[None, None, :]

        pad_starlet = int((starlet.shape[1] - 1) / 2)
        pad = int((kernel_size - 1) / 2)

        self.convs_A = nn.ModuleList()
        self.convs_S = nn.ModuleList()
        self.convs_starlet = nn.ModuleList()

        for i in range(n_scales - 1):
            filt = filters
            self.convs_A.append(nn.Conv2d(1, filt, kernel_size, bias=False, padding=pad * 2 ** i,
                                          dilation=2 ** i, padding_mode='reflect'))
            last_filt = filters if exact_rec else 1
            self.convs_S.append(nn.Conv2d(filt, last_filt, kernel_size, bias=False, padding=pad * 2 ** i,
                                          dilation=2 ** i, padding_mode='reflect', groups=last_filt))
            parametrize.register_parametrization(self.convs_A[i], "weight", NormalizeByL2())
            self.convs_starlet.append(nn.Conv2d(1, 1, starlet.shape[1], bias=False,
                                                padding=pad_starlet * 2 ** i, dilation=2 ** i,
                                                padding_mode='reflect'))
            self.convs_starlet[i].weight = torch.nn.Parameter(starlet2d, requires_grad=False)

        self.k_init = nn.Parameter(torch.ones(1), requires_grad=False)
        self.mininet = MiniNet(n=n_scales - 1)
        if pretrained is True:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                weights_path = os.path.join(current_dir, 'weights',
                                            'LEARNLET_FINAL_{}_{}_sc{}.pth'.format(filters, kernel_size, n_scales))
                self.load_state_dict(torch.load(weights_path, map_location=torch.device(device), weights_only=True))
                print("[info] Found pretrained weights for this configuration of parameters.")
            except (FileNotFoundError, RuntimeError):
                print("[info] Couldn't load weights for this configuration of parameters; continuing with random init.")

    def forward(self, x, sigma, no_coarse=False, no_neg=False, thresholds_hc=None, allscales=False):

        thresholds = self.mininet(self.k_init) * 5

        wt = []
        for i in range(len(self.convs_A)):
            x_in = x[:, i][:, None]
            x_a = self.convs_A[i](x_in)

            if self.exact_rec is True:
                x_a = torch.cat([x_in, x_a], dim=1)

            if thresholds_hc is not None:
                thresh = thresholds_hc[i]
            else:
                thresh = thresholds[i] * sigma

            expand_dims = x_a.dim() - thresh.dim()
            new_shape = thresh.shape + (1,) * expand_dims
            thresh_expanded = thresh.view(new_shape)

            if self.thresh == 'hard':
                if no_neg is False:
                    x_a_t = x_a * torch.sigmoid((torch.abs(x_a) - thresh_expanded) / 1e-3)
                else:
                    x_a_t = x_a * torch.sigmoid((x_a - thresh_expanded) / 1e-3)
            elif self.thresh == 'soft':
                x_a_t = torch.sign(x_a) * torch.relu(torch.abs(x_a) - thresh_expanded)
            else:
                print('Not implemented thresholding')

            if self.exact_rec is True:
                x_in_t = x_a_t[:, 0][:, None]
                x_s = self.convs_S[i](x_a_t[:, 1:])
                x_a_id = self.convs_A[i](x_in_t)
                x_s_id = self.convs_S[i](x_a_id)
                last_one = x_in_t - x_s_id.sum(dim=1, keepdim=True)
                x_s = torch.cat([last_one, x_s], dim=1)
                x_s = x_s.sum(dim=1, keepdim=True)
            else:
                x_s = self.convs_S[i](x_a_t)
            wt.append(x_s)

        if no_coarse is False:
            wt.append(x[:, i + 1][:, None])

        if allscales is False:
            rec = torch.cat(wt, dim=1).sum(dim=1, keepdim=True)
        else:
            rec = torch.cat(wt, dim=1)

        return rec
