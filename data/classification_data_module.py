import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.utils.data import random_split, DataLoader

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers):
        super(ClassificationDataModule, self).__init__()
        self.dataset = GenericClassificationDataset(torch_dataset=dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage: str) -> None:
        if stage=='fit':
            self.train, self.val = random_split(self.dataset, (0.8,0.2))
    
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)