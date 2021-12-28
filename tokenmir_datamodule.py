from abc import ABC, abstractmethod
import pytorch_lightning as pl


class TokenMIRDataModule(pl.LightningDataModule, ABC):

    @abstractmethod
    def get_target_distribution_weights(self):
        return None
