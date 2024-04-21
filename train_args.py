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
import argparse

pl.seed_everything(42, workers=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image classification models')
    parser.add_argument('--project_name', type=str, default='Involution', help='Name of the project')
    parser.add_argument('--dataset_name', type=str, default='Imagenet-Tiny', help='Name of the dataset')
    parser.add_argument('--reDSnet_type', type=int, default=50, help='Type of RedNet')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=200, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    parser.add_argument('--num_workers', type=int, default=40, help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--checkpoint_name', type=str, default='{epoch}-{step}', help='Checkpoint file name')
    parser.add_argument('--gradient_clip_val', type=float, default=0.8, help='Gradient clipping value')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--model', type=str, default='RedNet', choices=['RedNet', 'ResNet', 'DedNet', 'DenseNet'], help='Choose the model architecture')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    if args.model == 'RedNet':
        model_class = RedNetClassifier
    elif args.model == 'ResNet':
        model_class = ResNetClassifier
    elif args.model == 'DedNet':
        model_class = DedNetClassifier
    elif args.model == 'DenseNet':
        model_class = DenseNetClassifier
    else:
        raise ValueError("Invalid model choice. Choose from 'RedNet', 'ResNet', 'DedNet', 'DenseNet'.")

    configs = {
        'ProjectName': args.project_name,
        'model': model_class,
        'dataset_name': args.dataset_name,
        'dataset': ["tiny-imagenet-200/train","tiny-imagenet-200/val", "tiny-imagenet-200/test"],
        'ReDSnet_type': args.reDSnet_type,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'optimizer': SGD,
        'optimizer_kwargs': {
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'nesterov': args.nesterov
        },
        'num_workers': args.num_workers,
        'max_epochs': args.max_epochs,
        'lr_scheduler': CosineAnnealingLR,
        'lr_scheduler_kwargs': {
            'eta_min': 0,
            'T_max': args.max_epochs
        },
        'checkpoint_name': args.checkpoint_name,
        'gradient_clip_val': args.gradient_clip_val,
        'dropout': args.dropout
    }

    configs['experiment_name'] = f"model={args.model}-{configs['dataset_name']}-type={configs['ReDSnet_type']}-bs={configs['batch_size']}-{configs['optimizer']}-optkwargs={configs['optimizer_kwargs']}-dropout={configs['dropout']*100}-new_transforms-clip_val={configs['gradient_clip_val']}"
    configs['checkpoint_save_path'] = f"outputs/{configs['experiment_name']}"

    checkpoint_callback = ModelCheckpoint(dirpath=configs['checkpoint_save_path'], filename=configs['checkpoint_name'], monitor="val/epoch_accuracy", save_last=True, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    logger = WandbLogger(name=configs['experiment_name'], project=configs['ProjectName'])
    data_module = ClassificationDataModuleMulti(configs['dataset'], batch_size=configs['batch_size'], num_workers=configs['num_workers'])
    model = configs['model'](configs['ReDSnet_type'], configs['num_classes'], configs['optimizer'], configs['optimizer_kwargs'], configs['lr_scheduler'], configs['lr_scheduler_kwargs'], dropout=configs['dropout'])
    model.init_weights()
    trainer = pl.Trainer(max_epochs=configs['max_epochs'], check_val_every_n_epoch=2, callbacks=[checkpoint_callback, lr_monitor], logger=logger, gradient_clip_val=configs['gradient_clip_val'])
    trainer.fit(model, datamodule=data_module)
