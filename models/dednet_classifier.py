from typing import Any
from models.backbones.densenet import DenseNet
from models.prediction_heads.classifier import Classifier
from models.generic_model import GenericModel
import torch.nn as nn
import torch

class DedNetClassifier(GenericModel):
    def __init__(self, dednet_type, num_classes, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, dropout):
        super(DedNetClassifier, self).__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        types = {
                121: {
                        'growth_rate': 32,
                        'block_config': (6, 12, 24, 16),
                        'num_init_features': 64,
                },
                161: {
                        'growth_rate': 48,
                        'block_config': (6, 12, 36, 24),
                        'num_init_features': 96,
                },
                169: {
                        'growth_rate': 32,
                        'block_config': (6, 12, 32, 32),
                        'num_init_features': 64,
                },
                201: {
                        'growth_rate': 32,
                        'block_config': (6, 12, 48, 32),
                        'num_init_features': 64,
                    }
                }

        dednet_conf = types[dednet_type]
        self.backbone = DenseNet(
                dednet_conf['growth_rate'],
                dednet_conf['block_config'],
                dednet_conf['num_init_features'],
                drop_rate = dropout,
                num_classes = num_classes,
                is_dednet=True)


        self.test_epoch_metrics = {
            "accuracy": 0
        }
        self.val_epoch_metrics = {
            "accuracy": 0
        }
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def accuracy(self, true, pred, top_k=(1,)):

        max_k = max(top_k)
        batch_size = true.size(0)

        _, pred = pred.topk(max_k, 1)
        pred = pred.t()
        correct = pred.eq(true.view(1, -1).expand_as(pred))

        result = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            result.append(correct_k.div_(batch_size))

        return result[0]

    def get_test_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = outputs.shape[0]
        metrics = {}
        _, preds = outputs.max(1)
        true_positives = (preds==labels).sum()
        metrics["accuracy"] = self.accuracy(labels, outputs)#true_positives/batch_size
        return metrics
        
    def get_epoch_wise_metric_averages(self, metrics, epoch_metrics, batch_idx):
        epoch_metrics['accuracy'] = epoch_metrics['accuracy'] + (metrics['accuracy']-epoch_metrics['accuracy'])/(batch_idx+1)
        return epoch_metrics
    
    def reset_epoch_metrics(self, epoch_metrics):
        epoch_metrics['accuracy'] = 0
        return epoch_metrics
        
    def forward(self, images):
        x = self.backbone(images)
        return x
