import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.classification_data_module import ClassificationDataModule
from torchvision.datasets import ImageNet, CIFAR100, CIFAR100, MNIST, Caltech256
from models.rednet_classifier import RedNetClassifier
from models.resnet_classifier import ResNetClassifier
from models.dednet_classifier import DedNetClassifier
from models.densenet_classifier import DenseNetClassifier
from data.datasets.generic_classification_dataset import GenericClassificationDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, LinearLR, StepLR, PolynomialLR
pl.seed_everything(42, workers=True)
if __name__=="__main__":

    configs={
        'ProjectName': 'Involution',
        'model': RedNetClassifier,
        'dataset_name': 'Caltech256',
        'dataset': GenericClassificationDataset(torch_dataset=Caltech256(root=".")),
        'data_module_path': "data_module.pth",
        'ReDSnet_type': 26,
        'batch_size': 16,
        'num_classes': 257,
        'optimizer': SGD,
        'optimizer_kwargs': {
            'lr':0.1,
            'weight_decay':1e-4,
            "momentum": 0.9,
            'nesterov': True
        },
        'num_workers':40,
        'max_epochs': 150,
        'lr_scheduler': CosineAnnealingLR,
        'lr_scheduler_kwargs':{
            'T_max': 150,
        },
        'checkpoint_name':'{epoch}-{step}',
        "gradient_clip_val":1,
        'dropout': 0,
        'pretrained': 'last.ckpt'
    }
    configs['experiment_name'] = f"model={configs['model']}-{configs['dataset_name']}-type={configs['ReDSnet_type']}-bs={configs['batch_size']}-{configs['optimizer']}-optkwargs={configs['optimizer_kwargs']}-dropout={configs['dropout']*100}-new_transforms-clip_val={configs['gradient_clip_val']}-sdm"
    configs['checkpoint_save_path']=f"outputs/{configs['experiment_name']}"
    #configs["lr_scheduler_kwargs"]['T_max'] = configs["max_epochs"]*len(configs['dataset'])/configs["batch_size"]

    checkpoint_callback = ModelCheckpoint(dirpath=configs['checkpoint_save_path'], filename=configs['checkpoint_name'],monitor="val/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    logger = WandbLogger(name=configs['experiment_name'], project=configs['ProjectName'])
    if not configs["data_module_path"] is None:
        data_module = torch.load(configs["data_module_path"])
    else:
        data_module = ClassificationDataModule(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    data_module.batch_size = configs['batch_size']
    model = configs['model'](configs['ReDSnet_type'] ,configs['num_classes'], configs['optimizer'], configs['optimizer_kwargs'], configs['lr_scheduler'], configs['lr_scheduler_kwargs'], dropout = configs['dropout'])#.load_from_checkpoint("last.ckpt", optimizer = configs['optimizer'], optimizer_kwargs = configs['optimizer_kwargs'],lr_scheduler =  configs['lr_scheduler'],lr_scheduler_kwargs =  configs['lr_scheduler_kwargs'])
    if configs['pretrained'] is None:
        model.init_weights()
    else:
        model = configs['model'].load_from_checkpoint("last.ckpt", optimizer = configs['optimizer'], optimizer_kwargs = configs['optimizer_kwargs'],lr_scheduler =  configs['lr_scheduler'],lr_scheduler_kwargs =  configs['lr_scheduler_kwargs'])

    trainer = pl.Trainer(max_epochs=configs['max_epochs'], check_val_every_n_epoch=2,callbacks=[checkpoint_callback, lr_monitor], logger=logger, gradient_clip_val = configs['gradient_clip_val'] )
    trainer.test(model, datamodule=data_module)#, ckpt_path="last.ckpt")
    #torch.save(data_module,"data_module.pth")
