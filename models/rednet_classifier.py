from typing import Any
from models.backbones.resnet import ReDSNet
from models.prediction_heads.classifier import Classifier
from models.generic_model import GenericModel
import torch.nn as nn
import torch

class RedNetClassifier(GenericModel):
    def __init__(self, type, num_classes, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs):
        super(RedNetClassifier, self).__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        backbone = ReDSNet(type, is_rednet=True)
        self.add_module("backbone", backbone)
        self.add_module("adapool", nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("classifier", Classifier(512*backbone.expansion, num_classes))
        self.test_epoch_metrics = {
            "accuracy": 0
        }
        self.val_epoch_metrics = {
            "accuracy": 0
        }
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)

        
    def get_test_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = outputs.shape[0]
        metrics = {}
        _, preds = outputs.max(1)
        true_positives = (preds==labels).sum()
        metrics["accuracy"] = true_positives/batch_size
        return metrics
        
    def get_epoch_wise_metric_averages(self, metrics, epoch_metrics, batch_idx):
        epoch_metrics['accuracy'] = epoch_metrics['accuracy'] + (metrics['accuracy']-epoch_metrics['accuracy'])/(batch_idx+1)
        return epoch_metrics
    
    def reset_epoch_metrics(self, epoch_metrics):
        epoch_metrics['accuracy'] = 0
        return epoch_metrics
        
    def forward(self, images):
        x = self.backbone(images)
        x = self.adapool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    
    
