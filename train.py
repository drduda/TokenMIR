import fire
from fma_token_dataset import FMATokenDataModule
import pytorch_lightning as pl
from systems import MLMSystem, ClassificationSystem
import architectures
from pytorch_lightning.loggers import TensorBoardLogger
from spectrograms.data import FmaSpectrogramGenreDataModule

def classify_from_spectograms(fma_dir, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1, precision=32,
             name="default"):
    assert d_model % n_head == 0
    data_module =FmaSpectrogramGenreDataModule(fma_dir)
    logger = TensorBoardLogger("tb_log", name="specto/%s" % name)
    model = architectures.BERTWithoutEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=?, output_units=16, input_units=)
    import torch
    mir_system = ClassificationSystem(model=model, target_dist=torch.ones(16))
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

def classify_from_tokens(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1, precision=32,
             token_sequence_length=1024, name="default"):
    assert d_model % n_head == 0

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)
    logger = TensorBoardLogger("tb_log", name="tokens/%s" % name)
    model = architectures.BERTWithEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16)

    mir_system = ClassificationSystem(model=model, target_dist=data_module.target_distribution())
    trainer = pl.Trainer(logger=logger,
        max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus, precision=precision)
    trainer.fit(mir_system, data_module)


if __name__ == '__main__':
    fire.Fire()
