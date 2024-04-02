from typing import Any
from models.backbones.resnet import ReDSNet
from models.prediction_heads.classifier import Classifier
from models.generic_model import GenericModel
import torch.nn as nn
import torch
class RedNetClassifier(GenericModel):
    def __init__(self, num_classes):
        super(RedNetClassifier, self).__init__()
        backbone = ReDSNet(26, is_rednet=True)
        self.add_module("backbone", backbone)
        self.add_module("adapool", nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("classifier", Classifier(512*backbone.block.expansion, num_classes))
        self.test_epoch_metrics = {
            "accuracy": 0
        }
        self.val_epoch_metrics = {
            "accuracy": 0
        }

    def get_test_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        metrics = {}
        _, preds = outputs.max(1)
        true_positives = (preds==labels).sum()
        
    def forward(self, images):
        x = self.backbone(images)
        x = self.adapool(x)
        x = x.flatten()
        x = self.classifier(x)
        return x
    
    