from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class MultiStyleDataset(data.Dataset):
    def __init__(self, content_fname, styles_fpath, fine_size):
        super(MultiStyleDataset, self).__init__()
        assert is_image_file(content_fname), \
            '{} must be an image file'.format(content_fname)
        self.style_images_basedir = styles_fpath
        self.content_fname = content_fname
        self.style_image_fnames = [
            fname for fname in listdir(styles_fpath) if is_image_file(fname)]
        self.fine_size = fine_size
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Scale(fine_size),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self, index):
        styleImgPath = os.path.join(self.style_images_basedir, self.style_image_fnames[index])
        contentImg = default_loader(self.content_fname)
        styleImg = default_loader(styleImgPath)

        # resize
        if self.fine_size != 0:
            w, h = contentImg.size
            if(w > h):
                if w != self.fine_size:
                    neww = self.fine_size
                    newh = h * neww // w
                    contentImg = contentImg.resize((neww, newh))
                    styleImg = styleImg.resize((neww, newh))
            else:
                if h != self.fine_size:
                    newh = self.fine_size
                    neww = w * newh // h
                    contentImg = contentImg.resize((neww, newh))
                    styleImg = styleImg.resize((neww, newh))


        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        _out_fname_elems = self.content_fname.split('/')[-1].split('.')[:-1]
        out_fname = '{}_style_{}.jpg'.format('.'.join(_out_fname_elems), index)
        return contentImg.squeeze(0), styleImg.squeeze(0), out_fname

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.style_image_fnames)
