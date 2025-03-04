import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange
from sampling_alter import prior_guided_patch_sampling
import warnings
import time
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.*")

def split_dataset_oiqa(oiqa_csv_pach,seed=20):
    np.random.seed(seed)
    oiqa = pd.read_csv(oiqa_csv_pach)
    ref_name = []
    oiqa_length = len(oiqa)
    ref_column = oiqa['ref']
    for name in ref_column:
        if name not in ref_name:
            ref_name.append(name)
    l  = len(ref_name)
    split_index = int(np.floor(l * 0.8))
    train_ref = ref_name[:split_index]
    val_ref = ref_name[split_index:]
    train_name = []
    test_name = []
    for index,row in oiqa.iterrows():
        if row[0] in train_ref:
            train_name.append(row[1])
        if row[0] in val_ref:
            test_name.append(row[1])
    return train_name,test_name

class MyDataset(Dataset):
    def __init__(self, root, csv_path, model='train', transform=None, seed=100):
        self.seed = seed
        self.root = root
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        train_name, val_name = split_dataset_oiqa(csv_path, seed)

        self.name_to_label = {row['dis']: row['mos'] for _, row in self.data.iterrows()}

        if model == 'train':
            self.image_name = train_name
        elif model == 'test':
            self.image_name = val_name
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")

        self.transforms = transform

    def __getitem__(self, idx):
        image_name = self.image_name[idx]
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        label = self.name_to_label[image_name]

        if self.transforms:
            image = self.transforms(image)

        patches, position = prior_guided_patch_sampling(image, seed=self.seed)
        return patches, label
    def __len__(self):
        return len(self.image_name)




