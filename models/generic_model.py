from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
from torch.optim import Adam

class GenericModel(pl.LightningModule):
    def __init__(self):
        super(GenericModel, self).__init__()
        self.epoch_loss = 0
        self.test_epoch_metrics = {}
        self.val_epoch_metrics = {}
    
    def get_loss(self, outputs, labels):
        return NotImplementedError("Please Implement the Loss Function in Child Class")
    
    def get_test_metrics(self, outputs, labels):
        return NotImplementedError("Please Implement the Validation Metrics Function in Child Class")
    
    def get_epoch_wise_metric_averages(self, metrics, epoch_metrics, batch_idx):
        return NotImplementedError("Implement an Aggregating or Averaging methodology for testing metrics")
    
    def reset_epoch_metrics(self, epoch_metrics):
        return NotImplementedError("Implement a method to reset all the metrics per epoch that you used.")
    
    def configure_optimizers(self):
        return Adam(self.parameters(), 1e-5)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.get_loss(outputs=outputs, labels=labels)
        self.log("train/step_loss", loss.item())
        self.epoch_loss = self.epoch_loss + (loss.item()-self.epoch_loss)/(batch_idx+1)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        metrics = self.get_test_metrics(outputs, labels)
        for metric in metrics.keys():
            self.log(f"test/{metric}", metrics[metric])
        self.test_epoch_metrics = self.get_epoch_wise_metric_averages(metrics, self.test_epoch_metrics, batch_idx)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        metrics = self.get_test_metrics(outputs, labels)
        for metric in metrics.keys():
            self.log(f"val/{metric}", metrics[metric])
        self.val_epoch_metrics = self.get_epoch_wise_metric_averages(metrics, self.val_epoch_metrics, batch_idx)
        return metrics

    def on_train_epoch_end(self):
        self.log("train/epoch_loss", self.epoch_loss)
        self.epoch_loss = 0

    def on_test_epoch_end(self):
        for metric in self.epoch_metrics.keys():
            self.log(f"test/epoch_{metric}", self.epoch_metrics[metric])
        self.reset_epoch_metrics(self.epoch_metrics)
        