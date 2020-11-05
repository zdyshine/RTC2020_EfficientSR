import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np


def img_modcrop(image, modulo):
    """
    Modcrop an image

    Args:
        image: (array): write your description
        modulo: (todo): write your description
    """
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    """
    Convert tensor to a tensor.

    Args:
    """
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    """
    Determine if a file is an image is an image.

    Args:
        filename: (str): write your description
    """
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".JPG"])


def load_image(filepath):
    """
    Loads an image from disk.

    Args:
        filepath: (str): write your description
    """
    return Image.open(filepath).convert('RGB')


class DatasetFromFolderVal(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        """
        Initialize a list of images.

        Args:
            self: (todo): write your description
            hr_dir: (str): write your description
            lr_dir: (str): write your description
            upscale: (float): write your description
        """
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        """
        Get an index of an index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(img_modcrop(target, self.upscale))

        return input, target

    def __len__(self):
        """
        Returns the total number of filenames.

        Args:
            self: (todo): write your description
        """
        return len(self.lr_filenames)
