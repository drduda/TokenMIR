import pytorch_lightning as pl
import torch

MASK_TOKEN = 2048

class PretrainingSystem(pl.LightningModule):
    def __init__(self, module, masking_percentage):
        """
        The system for autoregressive training with masking.
        :param module: Deep learning module that is pretrained.
        """
        super().__init__()
        self.save_hyperparameters()
        self.module = module
        self.masking_percentage = masking_percentage

    def forward(self, x):
        return self.module(x)

    def training_step(self, batch, batch_idx):
        # Actual labels are discarded
        x, _ = batch
        y = x.detach().clone()

        # Mask the input
        rand = torch.rand(x.shape)
        mask_arr = rand < self.masking_percentage
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
    def __init__(self, output_logits, model_path=None, model=None):
        super().__init__()
        self.save_hyperparameters()

        if model_path:
            self.backbone = PretrainingSystem.load_from_checkpoint(model_path)
        else:
            self.backbone = model

        self.classifier = torch.nn.Sequential(torch.nn.Linear(2048, 200), torch.nn.ReLU(), torch.nn.Linear(200, output_logits))

    def configure_optimizers(self):
        # todo adjust optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        # Only take the first token for classification
        embedding = self.backbone(x)[:, 0, :]
        y_hat = self.classifier(embedding)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        return loss
