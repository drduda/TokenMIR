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
        # identifying number of correct predections in a given batch
        correct = y_hat.argmax(dim=1).eq(y).sum().item()

        # identifying total number of labels in a given batch
        total = len(y)

        # calculating the loss
        train_loss = torch.nn.functional.cross_entropy(y_hat, y.long(), weight=self.target_dist)

        # logs- a dictionary
        logs = {"train_loss": train_loss}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,

            # optional for batch logging purposes
            "log": logs,

            # info to be used at epoch end
            "correct": correct,
            "total": total
        }

        return batch_dictionary

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          correct / total,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)

        # identifying number of correct predections in a given batch
        correct = y_hat.argmax(dim=1).eq(y).sum().item()

        # identifying total number of labels in a given batch
        total = len(y)

        # calculating the loss
        train_loss = torch.nn.functional.cross_entropy(y_hat, y.long())

        # logs- a dictionary
        logs = {"train_loss": train_loss}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
            # optional for batch logging purposes
            "log": logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Val",
                                          correct / total,
                                          self.current_epoch)

    @staticmethod
    def accuracy(y_hat, y):
        pred = torch.max(y_hat, dim=1).indices
        acc = torch.sum(pred == y).item()/len(y)
        return acc
