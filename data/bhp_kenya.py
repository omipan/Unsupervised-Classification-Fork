import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
#from ct_utils import image_load_rgb,AddGaussianNoise
from matplotlib.image import imread
import csv

class BHPKenya(Dataset):
    """Biome health project dataset"""
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    base_folder = 'kenya18_data'
    train_list = ['train']
    val_list = ['val']
    test_list = ['test']

    meta = {
        'filename': 'class_to_target_idx.csv',
    }

    def __init__(self,
                 root=MyPath.db_root_dir('bhp_kenya'),
                 train=True,
                 transform=None,
                 download=False):


        super(BHPKenya, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        self.classes = ['wildebeest', 'gazelle_thomsons', 'shoats', 'cattle', 'zebra', 'impala',
                         'topi', 'warthog', 'hyena_spotted', 'giraffe', 'elephant', 'hare', 'dikdik',
                         'hippopotamus', 'jackal', 'gazelle_grants', 'baboon', 'buffalo', 'eland',
                         'mongoose_white_tailed', 'mongoose_banded', 'vervet_monkey', 'springhare',
                         'bateared_fox', 'waterbuck', 'hartebeest_cokes', 'domestic_dog', 'lion', 'aardvark',
                         'genet', 'serval', 'mongoose_other', 'porcupine', 'aardwolf', 'oribi',
                         'other_bird', 'ostrich', 'bustard_white_bellied', 'guineafowl']




        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.val_list

        self._load_meta()

        self.img_paths = []
        self.targets = []

        for folder in downloaded_list:
            folder_path = os.path.join(self.root, self.base_folder, folder)
            folder_files = os.listdir(folder_path)
            img_type = '.jpg'
            for file in folder_files:
                if file.endswith(img_type):
                    #img_data = imread(folder_path + '/' + file)
                    img_path = folder_path + '/' + file
                    species = '_'.join(file.split('_')[4:]).replace(img_type, '')
                    if species in self.classes:
                        self.img_paths.append(img_path)
                        self.targets.append(self.class_to_idx[species])

    def get_image(self, index):
        img_path = self.img_paths[index]
        img = imread(img_path)
        return img

    def _load_meta(self):
        path = os.path.join(self.root, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.')
        with open(path,mode='r') as infile:
            reader = csv.reader(infile)
            self.class_to_idx = {rows[0]: rows[1] for rows in reader}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img_path, target = self.img_paths[index], self.targets[index]


        img = imread(img_path)
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.idx_to_class[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': int(target), 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out