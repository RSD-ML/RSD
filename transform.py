import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
import numbers
import torch


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

def rr_train(resize_size=(224, 224)):
    return transforms.Compose([
    transforms.Scale(resize_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def rr_eval(resize_size=(224, 224)):
    return transforms.Compose([
        transforms.Scale(resize_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

