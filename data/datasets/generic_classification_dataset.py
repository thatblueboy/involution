from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as t
import torch
import numpy as np
from PIL import Image, ImageEnhance
'''
This code assumes the following folder structure
|- Main Dataset Folder
 |-- Class 1
   |- img1
   |- img2
   |- img3
   |- ...
   |- imgN
 |-- Class 2
   |- img1
   |- img2
   |- img3
   |- ...
   |- imgN
 |-- Class 3
   |- img1
   |- img2
   |- img3
   |- ...
   |- imgN
  ...
'''

class GenericClassificationDataset(Dataset):
    def __init__(self, dataset_path = None, torch_dataset = None, split = "train"):
        if dataset_path is None and torch_dataset is None:
            ValueError("Must be already implemented in pytorch or a custom dataset path with folder structure as given in this file.")

        self.transforms_train = t.Compose([
            #t.Resize(244, Image.LANCZOS),
            t.ToImage(),
            t.ToDtype(torch.uint8, scale=True),
            #t.CenterCrop(244),
            t.RandomHorizontalFlip(0.5),
            t.RandomVerticalFlip(0.5),
            t.RandomRotation(45),
            t.RandomAffine(45),
            #t.ColorJitter(),
            t.ToDtype(torch.float32, scale=True),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transforms_val = t.Compose([
             t.ToImage(),
             t.ToDtype(torch.uint8, scale=True),
             t.CenterCrop(244),
             t.ToDtype(torch.float32, scale=True),
             t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.split = split

        self.ds_type=0

        if not torch_dataset is None:
            self.dataset = torch_dataset
        
        if not dataset_path is None:
            self.dataset = ImageFolder(dataset_path)
            self.ds_type=1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transforms_train(self.dataset[index]) if self.split == "train" else self.transforms_val(self.dataset[index])
        

