import os.path
import warnings
import fire
from fma_token_dataset import FMATokenDataModule, CodebookDataModule
import pytorch_lightning as pl
from systems import MLMSystem, ClassificationSystem, MaskedSpectroSystem
import architectures
from pytorch_lightning.loggers import TensorBoardLogger
from spectrograms.data import FmaSpectrogramGenreDataModule


def classify_from_spectrograms(fma_dir, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1,
                               precision=32, name="default", snippet_length=1024, n_mels=128, n_fft=2048,
                               hop_length=512, fma_subset="medium"):
    assert d_model % n_head == 0
    fma_dir = os.path.expanduser(fma_dir)

    # Most values are taken from librosa.stft
    data_module = FmaSpectrogramGenreDataModule(
        fma_dir, fma_subset, n_fft=n_fft, hop_length=hop_length, sr=44100, batch_size=batch_size, file_ext=".mp3",
        snippet_length=snippet_length, save_specs=True, n_mels=n_mels
    )
    logger = TensorBoardLogger("tb_log", name="spectro/%s" % name)
    model = architectures.BERTWithoutEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=snippet_length, output_units=16, input_units=n_mels)
    mir_system = ClassificationSystem(model=model, target_dist=data_module.get_target_distribution_weights())
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)


def classify_from_tokens(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1, precision=32,
             token_sequence_length=1024, name="default"):
    assert d_model % n_head == 0
    ds_path = os.path.expanduser(ds_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)
    logger = TensorBoardLogger("tb_log", name="tokens/%s" % name)
    model = architectures.BERTWithEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16)

    mir_system = ClassificationSystem(model=model, target_dist=data_module.get_target_distribution_weights())
    trainer = pl.Trainer(logger=logger,
        max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus, precision=precision)
    trainer.fit(mir_system, data_module)

def pretrain_from_spectrograms(fma_dir, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, row_mask_length,
                               gpus=-1, precision=32, name="default", snippet_length=1024, n_mels=128, n_fft=2048,
                               hop_length=512, fma_subset="medium"):
    assert d_model % n_head == 0
    fma_dir = os.path.expanduser(fma_dir)

    # Most values are taken from librosa.stft
    data_module = FmaSpectrogramGenreDataModule(
        fma_dir, fma_subset, n_fft=n_fft, hop_length=hop_length, sr=44100, batch_size=batch_size, file_ext=".mp3",
        snippet_length=snippet_length, save_specs=True, n_mels=n_mels
    )
    logger = TensorBoardLogger("tb_log", name="pretrain_spectro/%s" % name)
    model = architectures.BERTWithoutEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=snippet_length, output_units=16, input_units=n_mels)
    mir_system = MaskedSpectroSystem(model=model, row_mask_length=row_mask_length)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

def finetune_from_spectrograms(fma_dir, backbone_path, batch_size, epochs, learning_rate, gpus=-1, precision=32,
                               name="default", snippet_length=1024, n_mels=128, n_fft=2048,
                               hop_length=512, fma_subset="medium"):
    fma_dir = os.path.expanduser(fma_dir)
    backbone_path = os.path.expanduser(backbone_path)

    # Most values are taken from librosa.stft #todo take values from pretraining
    data_module = FmaSpectrogramGenreDataModule(
        fma_dir, fma_subset, n_fft=n_fft, hop_length=hop_length, sr=44100, batch_size=batch_size, file_ext=".mp3",
        snippet_length=snippet_length, save_specs=True, n_mels=n_mels
    )
    logger = TensorBoardLogger("tb_log", name="finetune_spectro/%s" % name)

    mir_system = ClassificationSystem(backbone_path=backbone_path,
                                      target_dist=data_module.get_target_distribution_weights(),
                                      static_learning_rate=learning_rate,
                                      lr_schedule=False)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

def pretrain_from_tokens(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, masking_percentage,
                         gpus=-1, precision=32, token_sequence_length=1024, checkpoint_path=None, ds_size="medium", name="default"):
    ds_path = os.path.expanduser(ds_path)

    if checkpoint_path:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length, ds_size)
    logger = TensorBoardLogger("tb_log", name="pretrain_tokens/%s" % name)
    model = architectures.BERTWithEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16)

    pretrain_system = MLMSystem(model=model, masking_percentage=masking_percentage)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision, resume_from_checkpoint=checkpoint_path)
    trainer.fit(pretrain_system, data_module)

def finetune_from_tokens(ds_path, backbone_path, batch_size, epochs, learning_rate, gpus=-1, precision=32,
                         token_sequence_length=1024, ds_size="medium", class_weighting=True, name="default"):
    ds_path = os.path.expanduser(ds_path)
    backbone_path = os.path.expanduser(backbone_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length, ds_size=ds_size)
    logger = TensorBoardLogger("tb_log", name="finetune_tokens/%s" % name)

    if class_weighting:
        mir_system = ClassificationSystem(backbone_path=backbone_path,
                                          target_dist=data_module.get_target_distribution_weights(),
                                          static_learning_rate=learning_rate,
                                          lr_schedule=False)
    else:
        mir_system = ClassificationSystem(backbone_path=backbone_path,
                                          static_learning_rate=learning_rate,
                                          lr_schedule=False)

    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

def classify_from_codebooks(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, gpus=-1, precision=32,
             token_sequence_length=1024, name="default"):
    assert d_model % n_head == 0
    ds_path = os.path.expanduser(ds_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)
    logger = TensorBoardLogger("tb_log", name="codebooks/%s" % name)
    model = architectures.BERTWithCodebooks(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16, input_units=64)
    mir_system = ClassificationSystem(model=model, target_dist=data_module.get_target_distribution_weights())
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

def pretrain_from_codebooks(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, masking_percentage,
                         gpus=-1, precision=32, token_sequence_length=1024, checkpoint_path=None, name="default"):
    ds_path = os.path.expanduser(ds_path)

    if checkpoint_path:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)
    logger = TensorBoardLogger("tb_log", name="pretrain_codebooks/%s" % name)
    model = architectures.BERTWithCodebooks(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16, input_units=64)

    pretrain_system = MLMSystem(model=model, masking_percentage=masking_percentage)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision, resume_from_checkpoint=checkpoint_path)
    trainer.fit(pretrain_system, data_module)

def finetune_from_codebooks(ds_path, backbone_path, batch_size, epochs, learning_rate, gpus=-1, precision=32,
                         token_sequence_length=1024, name="default"):
    ds_path = os.path.expanduser(ds_path)
    backbone_path = os.path.expanduser(backbone_path)

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)
    logger = TensorBoardLogger("tb_log", name="finetune_codebooks/%s" % name)

    mir_system = ClassificationSystem(backbone_path=backbone_path,
                                      target_dist=data_module.get_target_distribution_weights(),
                                      static_learning_rate=learning_rate,
                                      lr_schedule=False)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full', gpus=gpus,
                         precision=precision)
    trainer.fit(mir_system, data_module)

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=UserWarning)
    fire.Fire()
