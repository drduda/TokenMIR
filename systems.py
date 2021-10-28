import pytorch_lightning as pl
import torch

MASK_TOKEN = 2048
CLS_TOKEN = 2049

class MLMSystem(pl.LightningModule):
    def __init__(self, backbone, masking_percentage):
        """
        The system for autoregressive training with masking.
        :param backbone: Deep learning backbone that is pretrained.
        """
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.masking_percentage = masking_percentage

    def forward(self, x):
        # First token is CLS_TOKEN
        x[:, 0] = CLS_TOKEN

        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Actual labels are discarded
        x, _ = batch
        y = x.detach().clone()

        # Mask the input
        rand = torch.rand(x.shape)
        mask_arr = rand < self.masking_percentage
        # CLS_TOKEN should not be masked
        mask_arr[:, 0] = False
        x[mask_arr] = MASK_TOKEN

        y_hat = self(x)
        loss = self._loss(y_hat, y, mask_arr)
        return loss

    def configure_optimizers(self):
        #todo adjust optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def _loss(y_hat, y, mask_arr):
        # Adjust shape and dtypes
        y_hat = torch.flatten(y_hat, end_dim=1)
        mask_arr = torch.flatten(mask_arr, end_dim=1)
        y = torch.flatten(y).long()

        loss = torch.nn.functional.cross_entropy(y_hat, y, reduction="none")

        # Only the masked tokens are important for the loss
        loss = torch.mean(loss * mask_arr)
        return loss


class ClassificationSystem(pl.LightningModule):
    def __init__(self, classifier, model_path=None, model=None):
        super().__init__()
        self.save_hyperparameters()

        if model_path:
            self.backbone = MLMSystem.load_from_checkpoint(model_path)
        else:
            self.backbone = model

        self.classifier = classifier

    def configure_optimizers(self):
        # todo adjust optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        # First token is CLS_TOKEN
        x[:, 0] = CLS_TOKEN

        # Only take the first token for classification
        embedding = self.backbone(x)[:, 0, :]
        y_hat = self.classifier(embedding)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        return loss
