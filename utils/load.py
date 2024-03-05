import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

p = transforms.Compose([transforms.Resize((384,384))])

def get_ids(dir):
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=1):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, j) for id in ids for j in range(n))


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_nir = Image.open(dir + 'train/' + id[:] + suffix)
        img_nir = p(img_nir)
        img_nir = np.array(img_nir)
        img_nir = img_nir[np.newaxis, ...] 
        img_n = np.concatenate((img_nir), axis=0) 
        yield img_n

def to_cropped_mask(ids, dir, suffix):
    # From a list of tuples, returns the correct cropped img
    for id in ids:
        img = Image.open(dir + id + suffix)
        img = p(img)
        im = np.array(img)
        im = im/255 
        yield im

def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.jpg')
    masks = to_cropped_mask(ids, dir_mask, '.png')

    return zip(imgs, masks)

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    im = p(im)
    mask = Image.open(dir_mask + id + '.png')
    mask = p(mask)
    return np.array(im), np.array(mask)

def get_test_img(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_nir = Image.open(dir + 'test/' +  id[:] + suffix)
        img_nir = p(img_nir)
        img_nir = np.array(img_nir)
        img_nir = img_nir[np.newaxis, ...]
        img_n = np.concatenate((img_nir), axis=0)
        yield [img_n, id]


