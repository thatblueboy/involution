import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.classification_data_module import ClassificationDataModule
from torchvision.datasets import ImageNet, CIFAR100, CIFAR100, MNIST
from models.rednet_classifier import RedNetClassifier
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, PolynomialLR
pl.seed_everything(42, workers=True)
if __name__=="__main__":

    configs={
        'ProjectName': 'Involution',
	    'experiment_name': "New_expmt",
        'dataset': CIFAR100(root=".", train=True, download=True),
        'ReDSnet_type':50,
        'batch_size': 16,
        'num_classes': 200,
        'optimizer': Adam,
        'optimizer_kwargs': {
            'lr':1e-5,
            'weight_decay':0.0,
        },
        'num_workers':16,
	    'max_epochs': 10,
        'lr_scheduler': CosineAnnealingLR,
        'lr_scheduler_kwargs':{
            'T_max':10,
            'eta_min':1e-6
        },
        'checkpoint_name':'{epoch}-{step}'
    }
    configs['checkpoint_save_path']=f"outputs/{configs['experiment_name']}"

    checkpoint_callback = ModelCheckpoint(dirpath=configs['checkpoint_save_path'], filename=configs['checkpoint_name'],monitor="val/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    logger = WandbLogger(name=configs['experiment_name'], project=configs['ProjectName'])
    data_module = ClassificationDataModule(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    model = RedNetClassifier(configs['ReDSnet_type'] ,configs['num_classes'], configs['optimizer'], configs['optimizer_kwargs'], configs['lr_scheduler'], configs['lr_scheduler_kwargs'])
    trainer = pl.Trainer(max_epochs=configs['max_epochs'], check_val_every_n_epoch=2,callbacks=[checkpoint_callback, lr_monitor], logger=logger)
    trainer.fit(model, datamodule=data_module)
    