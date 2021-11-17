import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics

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
        #todo adjust for new api of archtitecture
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

        #todo is the loss correct
        loss = torch.nn.functional.cross_entropy(y_hat, y, reduction="none")

        # Only the masked tokens are important for the loss
        loss = torch.mean(loss * mask_arr)
        return loss


class ClassificationSystem(pl.LightningModule):
    def __init__(self, model_path=None, model=None, target_dist=None):
        super().__init__()
        self.save_hyperparameters()
        self.warmup_steps = 8000
        self.target_dist = target_dist

        if model_path:
            self.BERT = MLMSystem.load_from_checkpoint(model_path)
        else:
            self.BERT = model


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=[.9, .999])
        lr_func = lambda step: self.BERT.d_model**-.5 * \
                               min((step+1)**-.5, (step+1) * (self.warmup_steps**-1.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "step"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def forward(self, x):
        # First token is CLS_TOKEN
        x[:, 0] = CLS_TOKEN

        return self.BERT(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long(), weight=self.target_dist.type_as(y_hat))
        return {'loss': loss, 'preds': y_hat, 'target': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long(), weight=self.target_dist.type_as(y_hat))
        return {'loss': loss, 'preds': y_hat, 'target': y}

    def _at_epoch_end(self, outputs, stage=None):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        loss = torch.Tensor([tmp['loss'] for tmp in outputs])

        loss = torch.mean(loss).item()
        self.logger.experiment.add_scalar("Loss/%s" % stage, loss, self.current_epoch)

        num_classes = self.BERT.output_units

        f1 = torchmetrics.functional.classification.f1(preds, targets, average='macro', num_classes=num_classes)
        self.logger.experiment.add_scalar("F1_macro/%s" % stage, f1, self.current_epoch)

        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=num_classes)

        # Plot confusion matrix
        df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(num_classes), columns=range(num_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='g').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix/%s" % stage, fig_, self.current_epoch)

    def training_epoch_end(self, outputs):
        self._at_epoch_end(outputs, 'Train')

    def validation_epoch_end(self, outputs):
        self._at_epoch_end(outputs, 'Val')
