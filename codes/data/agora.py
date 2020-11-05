import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common

def default_loader(path):
    """
    Return the default loader.

    Args:
        path: (str): write your description
    """
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    """
    Load a npy. load.

    Args:
        path: (str): write your description
    """
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy', '.JPG'
]

def is_image_file(filename):
    """
    Determine if a file is an image.

    Args:
        filename: (str): write your description
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """
    Make a list of images.

    Args:
        dir: (str): write your description
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class agora(data.Dataset):
    def __init__(self, opt):
        """
        Initialize a set of the image.

        Args:
            self: (todo): write your description
            opt: (dict): write your description
        """
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext   # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        """
        Sets the filesystem.

        Args:
            self: (todo): write your description
            dir_data: (str): write your description
        """
        self.root = dir_data + '/train_data'
        self.dir_hr = os.path.join(self.root, 'hr_npy')
        self.dir_lr = os.path.join(self.root, 'lr_npy/x' + str(self.scale))

    def __getitem__(self, idx):
        """
        Return a tensor for a tensor

        Args:
            self: (todo): write your description
            idx: (list): write your description
        """
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        """
        Return the number of rows.

        Args:
            self: (todo): write your description
        """
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        """
        Return index of the index

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        """
        Get image size of image.

        Args:
            self: (todo): write your description
            img_in: (todo): write your description
            img_tar: (str): write your description
        """
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        """
        Scan the directory.

        Args:
            self: (todo): write your description
        """
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        print(len(list_hr), len(list_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        """
        Load the image file

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr
