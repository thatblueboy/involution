import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from data.classification_data_module import ClassificationDataModule
from torchvision.datasets import ImageNet, CIFAR100, CIFAR100, MNIST
from models.rednet_classifier import RedNetClassifier
from data.datasets.generic_classification_dataset import GenericClassificationDataset
pl.seed_everything(42, workers=True)
if __name__=="__main__":

    configs={
        'dataset': GenericClassificationDataset(dataset_path="tiny-imagenet-200"),
        'ReDSnet_type':50,
        'batch_size': 8,
        'num_classes': 200,
        'num_workers':8
    }
    checkpoint_callback = ModelCheckpoint(monitor="test/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    
    data_module = ClassificationDataModule(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    model = RedNetClassifier(configs['ReDSnet_type'] ,configs['num_classes'])
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=2,callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data_module)
    