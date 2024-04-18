import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.classification_data_module import ClassificationDataModule
from torchvision.datasets import ImageNet, CIFAR100, CIFAR100, MNIST, Caltech256
from models.rednet_classifier import RedNetClassifier
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, PolynomialLR
import argparse
pl.seed_everything(42, workers=True)

def get_args():
    parser = argparse.ArgumentParser(description='Train RedNet Classifier')
    parser.add_argument('--rednet_type', type=int, default=26, help='Type of RedNet')
    parser.add_argument('--max_epochs', type=int, default=25, help='Maximum number of epochs')
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()

    configs={
        'ProjectName': 'Involution',
        'isRedNet': False,
        'dataset_name': 'imagenet_tiny',
        'dataset': GenericClassificationDataset(dataset_path="tiny-imagenet-200"),
        'ReDSnet_type':args.rednet_type,
        'batch_size': 96,
        'num_classes': 200,
        'optimizer': SGD,
        'optimizer_kwargs': {
            'lr':0.01,
            'weight_decay':0.0,
        },
        'num_workers':4,
        'max_epochs': args.max_epochs,
        'lr_scheduler': None,
        'lr_scheduler_kwargs':{
            'eta_min':0.0,
            'T_max': 130
        },
        'checkpoint_name':'{epoch}-{step}',
        "gradient_clip_val":None
    }
    configs['experiment_name'] = f"{configs['dataset_name']}-type={configs['ReDSnet_type']}-bs={configs['batch_size']}-{configs['optimizer']}-lr={configs['optimizer_kwargs']['lr']}"
    configs['checkpoint_save_path']=f"outputs/{configs['experiment_name']}"
    #configs["lr_scheduler_kwargs"]['T_max'] = configs["max_epochs"]*len(configs['dataset'])/configs["batch_size"]

    checkpoint_callback = ModelCheckpoint(dirpath=configs['checkpoint_save_path'], filename=configs['checkpoint_name'],monitor="val/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    logger = WandbLogger(name=configs['experiment_name'], project=configs['ProjectName'])
    data_module = ClassificationDataModule(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    model = RedNetClassifier(configs['ReDSnet_type'] ,configs['num_classes'], configs['optimizer'], configs['optimizer_kwargs'], configs['lr_scheduler'], configs['lr_scheduler_kwargs'], configs['isRedNet'])
    model.init_weights()
    trainer = pl.Trainer(max_epochs=configs['max_epochs'], check_val_every_n_epoch=2,callbacks=[checkpoint_callback, lr_monitor], logger=logger, gradient_clip_val = configs['gradient_clip_val'] )
    trainer.fit(model, datamodule=data_module)
    
