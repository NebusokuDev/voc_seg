from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam


class SegmentationModel(LightningModule):
    def __init__(self, model: Module, criterion: Module = None):
        super().__init__()
        self.model = model
        self.criterion = criterion or BCEWithLogitsLoss()

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)

        return {"test_loss": loss}
