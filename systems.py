import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics
from architectures import N_TOKENS, CLS_TOKEN, MASK_TOKEN, N_SPECIAL_TOKENS




class MySystem(pl.LightningModule):
    """
    Superclass for the other systems.
    """
    def __init__(self, lr_schedule):
        super().__init__()
        self.warmup_steps = 8000
        self.lr_schedule = lr_schedule

    def forward(self, x):
        # First token is CLS_TOKEN
        if x.ndim == 2:
            x[:, 0] = CLS_TOKEN

        return self.BERT(x)

    def _lr_func(self, step):
        return self.BERT.d_model ** -.5 * min((step + 1) ** -.5, (step + 1) * (self.warmup_steps ** -1.5))

    def configure_optimizers(self):
        if self.lr_schedule:
            optimizer = torch.optim.Adam(self.parameters(), betas=[.9, .999])

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_func)
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
        else:
            #todo make adjustable
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    def _at_epoch_end(self, outputs, stage=None):
        # Loss
        loss = torch.Tensor([tmp['loss'] for tmp in outputs])
        loss = torch.mean(loss).item()
        self.logger.experiment.add_scalar("Loss/%s" % stage, loss, self.current_epoch)

    def training_epoch_end(self, outputs):
        self._at_epoch_end(outputs, 'Train')

    def validation_epoch_end(self, outputs):
        self._at_epoch_end(outputs, 'Val')


class MLMSystem(MySystem):
    def __init__(self, model, masking_percentage):
        """
        The system for autoregressive training with masking.
        :param backbone: Deep learning backbone that is pretrained with MLMSystem.

        Cite BERT: 15% of tokens are chosen. From these chosen tokens 80% are masked,
        10% random, 10% remain unchanged
        """
        super().__init__(lr_schedule=True)
        self.save_hyperparameters()
        self.BERT = model
        self.masking_percentage = masking_percentage

        self.masked_token_share = 0.8
        self.random_token_share = .1

        assert self.masked_token_share + self.random_token_share <= 1

    def step(self, batch, batch_idx):
        # Actual labels are discarded
        x, _ = batch
        y = x.detach().clone()

        # Select labels
        rand = torch.rand(x.shape).type_as(x)
        selected_arr = rand < self.masking_percentage
        # CLS_TOKEN should not be label
        selected_arr[:, 0] = False

        # Chose masked token, random token and implicit unchanged
        rand = torch.rand(x.shape).type_as(x)
        masked_token_arr = rand < self.masked_token_share
        random_token_arr = rand > (self.masked_token_share + self.random_token_share)

        # Combine with selected
        masked_token_arr = torch.logical_and(selected_arr, masked_token_arr)
        random_token_arr = torch.logical_and(selected_arr, random_token_arr)

        # Change input
        x[masked_token_arr] = MASK_TOKEN
        x[random_token_arr] = torch.randint_like(x[random_token_arr], low=0, high=N_TOKENS)

        _, y_hat = self(x)
        loss = self._loss(y_hat, y, selected_arr)
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    @staticmethod
    def _loss(y_hat, y, selected_token_arr):
        # Adjust shape and dtypes
        y_hat = torch.flatten(y_hat, end_dim=1)
        selected_token_arr = torch.flatten(selected_token_arr, end_dim=1)
        y = torch.flatten(y).long()

        #todo is the loss correct
        loss = torch.nn.functional.cross_entropy(y_hat, y, reduction="none")

        # Only the masked tokens are important for the loss
        loss = torch.mean(loss * selected_token_arr)
        return loss


class ClassificationSystem(MySystem):
    def __init__(self, backbone_path=None, model=None, target_dist=None, lr_schedule=True):
        super().__init__(lr_schedule=lr_schedule)
        self.save_hyperparameters()
        self.target_dist = target_dist

        if backbone_path:
            self.BERT = MLMSystem.load_from_checkpoint(backbone_path).BERT
        else:
            self.BERT = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long(), weight=self.target_dist.type_as(y_hat))
        return {'loss': loss, 'preds': y_hat.detach(), 'target': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long(), weight=self.target_dist.type_as(y_hat))
        return {'loss': loss, 'preds': y_hat.detach(), 'target': y}

    def _at_epoch_end(self, outputs, stage=None):
        super()._at_epoch_end(outputs, stage)

        # Accuracy
        num_classes = self.BERT.output_units

        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        f1 = torchmetrics.functional.classification.f1(preds, targets, average='macro', num_classes=num_classes)
        self.logger.experiment.add_scalar("F1_macro/%s" % stage, f1, self.current_epoch)

        # Confusion matrix
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=num_classes)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(num_classes), columns=range(num_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='g').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix/%s" % stage, fig_, self.current_epoch)
