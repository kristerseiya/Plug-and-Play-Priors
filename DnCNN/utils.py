
import torch
from torch import nn
from . import model


def load_dncnn(model_path, device=None):

    n_channels = 1        # fixed for grayscale image

    nb = 17               # fixed

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    state_dict = torch.load(model_path, map_location=device)
    nb = len(state_dict.keys()) // 2
    if state_dict['model.0.weight'].size(1) == 3:
        n_channels = 3
    else:
        n_channels = 1

    net = model.DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    net.load_state_dict(state_dict, strict=True)
    if not torch.cuda.is_available():
        torch.set_flush_denormal(True)
    net.eval()
    for k, v in net.named_parameters():
        v.requires_grad = False
    net = net.move(device)

    return net

def merge_bn(model):
    ''' Kai Zhang, 11/Jan/2019.
    merge all 'Conv+BN' (or 'TConv+BN') into 'Conv' (or 'TConv')
    based on https://github.com/pytorch/pytorch/pull/901
    '''
    if type(model) == nn.Sequential:
        new_keys = list()
        count = 0
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        elif type(model) == nn.Sequential:
            new_keys.append(str(count))
            count = count + 1
        prev_m = m
        merge_bn(m)
    if type(model) == nn.Sequential:
        for nk, ok in zip(new_keys, list(model._modules.keys())):
            model._modules[nk] = model._modules.pop(ok)
