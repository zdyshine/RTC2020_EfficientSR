import random
import torch
import numpy as np
import skimage.color as sc


def get_patch(*args, patch_size, scale):
    """
    Return a set of random size of size

    Args:
        patch_size: (int): write your description
        scale: (float): write your description
    """
    ih, iw = args[0].shape[:2]

    tp = patch_size  # target patch (HR)
    ip = tp // scale  # input patch (LR)

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]  # results
    return ret


def set_channel(*args, n_channels=3):
    """
    Set the channel.

    Args:
        n_channels: (int): write your description
    """
    def _set_channel(img):
        """
        Set the rgb image to a channel to a channel.

        Args:
            img: (array): write your description
        """
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range):
    """
    Convert rgb tensor to a tensor.

    Args:
        rgb_range: (todo): write your description
    """
    def _np2Tensor(img):
        """
        Convert tensor to tensor

        Args:
            img: (array): write your description
        """
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    """
    Augment of the image.

    Args:
        hflip: (todo): write your description
        rot: (todo): write your description
    """
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        """
        Transpose of - if the given an image.

        Args:
            img: (array): write your description
        """
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

