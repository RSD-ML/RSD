import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image

def make_dataset_r(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in xrange(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([float(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

class ImageList_r(object):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset_r(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

