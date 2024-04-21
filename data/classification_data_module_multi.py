import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.utils.data import random_split, DataLoader, Dataset

class ClassificationDataModuleMulti(pl.LightningDataModule):
    def __init__(self, datasets, batch_size, num_workers):
        super(ClassificationDataModuleMulti, self).__init__()
        self.train = GenericClassificationDataset(dataset_path=datasets[0], split = "train")
        self.val = GenericClassificationDataset(dataset_path=datasets[1], split = "val")
        self.test = GenericClassificationDataset(dataset_path=datasets[2], split = "test")
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage: str) -> None:
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
