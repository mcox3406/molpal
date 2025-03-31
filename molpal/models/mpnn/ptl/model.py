import logging
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from torch.optim import Adam
from torch.nn import functional as F

from molpal.models import mpnn
from molpal.models.chemprop.nn_utils import NoamLR

logging.getLogger("lightning").setLevel(logging.FATAL)


class LitMPNN(pl.LightningModule):
    """A message-passing neural network base class"""

    def __init__(self, config: Dict):
        super().__init__()
        
        # makes the models state dict checkpointable with Lightning
        self.save_hyperparameters()
        
        self.mpnn = config["model"]
        self.uncertainty = config["uncertainty"]
        self.dataset_type = config["dataset_type"]
        
        self.loss_func = mpnn.utils.get_loss_func(self.dataset_type, self.uncertainty)
        
        self.batch_size = config.get("batch_size", 50)
        self.warmup_epochs = config.get("warmup_epochs", 2.0)
        self.max_epochs = config.get("max_epochs", 50)
        self.init_lr = config.get("init_lr", 1e-4)
        self.max_lr = config.get("max_lr", 1e-3)
        self.final_lr = config.get("final_lr", 1e-4)
        
    def forward(self, x):
        return self.mpnn(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        mask = ~torch.isnan(y)
        y = torch.nan_to_num(y, nan=0.0)
        class_weights = torch.ones_like(y)
        
        y_hat = self.mpnn(x)
        
        if self.uncertainty == "mve":
            y_hat_mean = y_hat[:, 0::2]
            y_hat_var = y_hat[:, 1::2]
            loss = self.loss_func(y_hat_mean, y_hat_var, y)
        else:
            loss = self.loss_func(y_hat, y) * class_weights * mask
            
        loss = loss.sum() / mask.sum()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.mpnn(x)
        
        if self.uncertainty == "mve":
            y_hat = y_hat[:, 0::2]
            
        loss = F.mse_loss(y_hat, y, reduction="mean")
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.init_lr)
        
        num_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = num_steps / self.max_epochs
        
        scheduler = NoamLR(
            optimizer=optimizer,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.max_epochs],
            steps_per_epoch=steps_per_epoch,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.train_dataloader)

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
