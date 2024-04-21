import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.classification_data_module import ClassificationDataModule
from data.classification_data_module_multi import ClassificationDataModuleMulti
from torchvision.datasets import ImageNet, CIFAR100, CIFAR100, MNIST, Caltech256
from models.rednet_classifier import RedNetClassifier
from models.resnet_classifier import ResNetClassifier
from models.dednet_classifier import DedNetClassifier
from models.densenet_classifier import DenseNetClassifier
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, PolynomialLR
pl.seed_everything(42, workers=True)
if __name__=="__main__":

    configs={
        'ProjectName': 'Involution',
        'model': RedNetClassifier,
        'dataset_name': 'Imagenet-Tiny',
        'dataset': ["tiny-imagenet-200/train","tiny-imagenet-200/val", "tiny-imagenet-200/test"],
        'ReDSnet_type': 50,
        'batch_size': 256,
        'num_classes': 200,
        'optimizer': SGD,
        'optimizer_kwargs': {
            'lr':0.8,
            'weight_decay': 0.0,
            "momentum": 0.9,
            'nesterov': True
        },
        'num_workers':40,
        'max_epochs': 200,
        'lr_scheduler': CosineAnnealingLR,
        'lr_scheduler_kwargs':{
            'eta_min': 0,
            'T_max': 200
        },
        'checkpoint_name':'{epoch}-{step}',
        "gradient_clip_val":0.8,
        'dropout': 0.
    }
    configs['experiment_name'] = f"model={configs['model']}-{configs['dataset_name']}-type={configs['ReDSnet_type']}-bs={configs['batch_size']}-{configs['optimizer']}-optkwargs={configs['optimizer_kwargs']}-dropout={configs['dropout']*100}-new_transforms-clip_val={configs['gradient_clip_val']}"
    configs['checkpoint_save_path']=f"outputs/{configs['experiment_name']}"
    #configs["lr_scheduler_kwargs"]['T_max'] = configs["max_epochs"]*len(configs['dataset'])/configs["batch_size"]

    checkpoint_callback = ModelCheckpoint(dirpath=configs['checkpoint_save_path'], filename=configs['checkpoint_name'],monitor="val/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    logger = WandbLogger(name=configs['experiment_name'], project=configs['ProjectName'])
    data_module = ClassificationDataModuleMulti(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    model = configs['model'](configs['ReDSnet_type'] ,configs['num_classes'], configs['optimizer'], configs['optimizer_kwargs'], configs['lr_scheduler'], configs['lr_scheduler_kwargs'], dropout = configs['dropout'])#.load_from_checkpoint("last.ckpt", optimizer = configs['optimizer'], optimizer_kwargs = configs['optimizer_kwargs'],lr_scheduler =  configs['lr_scheduler'],lr_scheduler_kwargs =  configs['lr_scheduler_kwargs'])
    model.init_weights()
    trainer = pl.Trainer(max_epochs=configs['max_epochs'], check_val_every_n_epoch=2,callbacks=[checkpoint_callback, lr_monitor], logger=logger, gradient_clip_val = configs['gradient_clip_val'] )
    trainer.fit(model, datamodule=data_module)
    #print(model)
