from pathlib import Path

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

from transforms import Compose


class VOCDatamodule(LightningDataModule):
    def __init__(self, root="./data", batch_size=32):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.transform = Compose([

        ])

    def prepare_data(self) -> None:
        try:
            if not (self.root / "VOCdevkit").exists():
                VOCSegmentation(self.root, download=True, image_set="train")
                VOCSegmentation(self.root, download=True, image_set="val")
                VOCSegmentation(self.root, download=True, image_set="test", year="2007")
        except Exception as e:
            print(f"Error during dataset preparation: {e}")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._train_dataset = VOCSegmentation(self.root, image_set="train", transform=self.transform)
            self._val_dataset = VOCSegmentation(self.root, image_set="val", transform=self.transform)
        if stage == "test" or stage is None:
            self._test_dataset = VOCSegmentation(self.root, image_set="test", year="2007", transform=self.transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._test_dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    voc = VOCDatamodule()
    voc.prepare_data()
    voc.setup()
    dataloader = voc.train_dataloader()

    for batch in enumerate(dataloader):
        print(batch)
