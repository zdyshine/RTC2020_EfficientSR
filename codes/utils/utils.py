from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import numpy as np
import os
import torch
from collections import OrderedDict
import torch.nn as nn

def compute_psnr(im1, im2):
    """
    Compute the psnr.

    Args:
        im1: (todo): write your description
        im2: (str): write your description
    """
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    """
    R computes the covariance matrix.

    Args:
        im1: (float): write your description
        im2: (array): write your description
    """
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def shave(im, border):
    """
    Shave the border of a given image.

    Args:
        im: (int): write your description
        border: (int): write your description
    """
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    """
    Modcrop an image.

    Args:
        im: (array): write your description
        modulo: (todo): write your description
    """
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    """
    Get a list of files.

    Args:
        path: (str): write your description
        ext: (str): write your description
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convert_shape(img):
    """
    Convert image to shape.

    Args:
        img: (array): write your description
    """
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    """
    Quantize an image.

    Args:
        img: (array): write your description
    """
    return img.clip(0, 255).round().astype(np.uint8)


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Convert tensor to numpy. ndarray.

    Args:
        tensor: (todo): write your description
        out_type: (str): write your description
        np: (todo): write your description
        uint8: (todo): write your description
        min_max: (float): write your description
    """
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    """
    Convert tensor to numpy array.

    Args:
        tensor: (todo): write your description
    """
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    """
    Adjusts learning rate of the optimizer.

    Args:
        optimizer: (todo): write your description
        epoch: (int): write your description
        step_size: (int): write your description
        lr_init: (todo): write your description
        gamma: (todo): write your description
    """
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):
    """
    Loads a dictionary from a dictionary.

    Args:
        path: (str): write your description
    """

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
        """
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        """
        Forward computation

        Args:
            self: (todo): write your description
            X: (todo): write your description
            Y: (todo): write your description
        """
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)

        loss = torch.sum(error)
        return loss
