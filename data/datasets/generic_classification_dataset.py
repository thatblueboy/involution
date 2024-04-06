from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as t
import torch
import numpy as np
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
    def __init__(self, dataset_path = None, torch_dataset = None):
        if dataset_path is None and torch_dataset is None:
            ValueError("Must be already implemented in pytorch or a custom dataset path with folder structure as given in this file.")

        self.transforms = t.Compose([
            t.ToTensor(),
            t.ToDtype(torch.uint8, scale=True),
            t.RandomHorizontalFlip(0.5),
            t.ToDtype(torch.float32, scale=True),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.ds_type=0

        if not torch_dataset is None:
            self.dataset = torch_dataset
        
        if not dataset_path is None:
            self.dataset = ImageFolder(dataset_path, self.transforms)
            self.ds_type=1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        print(np.asarray(self.dataset[index][0]).shape)
        return self.transforms(self.dataset[index])
        

