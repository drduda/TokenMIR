import fire
from fma_token_dataset import FMATokenDataModule
import pytorch_lightning as pl
from systems import MLMSystem, ClassificationSystem
import architectures
from pytorch_lightning.loggers import TensorBoardLogger
from spectrograms.data import FmaSpectrogramGenreDataModule


def classify_from_spectrograms(fma_dir, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1, precision=32,
             name="default"):
    assert d_model % n_head == 0
    # Most values are taken from librosa.stft
    data_module = FmaSpectrogramGenreDataModule(
        fma_dir, "small", n_fft=2048, hop_length=512, sr=44100, batch_size=batch_size, file_ext=".mp3",
        snippet_length=1024, save_specs=True
    )
    logger = TensorBoardLogger("tb_log", name="spectro/%s" % name)
    model = architectures.BERTWithoutEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=500, output_units=16, input_units=100)
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

    mir_system = ClassificationSystem(model=model, target_dist=data_module.get_target_distribution_weights())
    trainer = pl.Trainer(logger=logger,
        max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus, precision=precision)
    trainer.fit(mir_system, data_module)


if __name__ == '__main__':
    fire.Fire()
