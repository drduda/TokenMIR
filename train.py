import fire
from fma_token_dataset import FMATokenDataModule
import pytorch_lightning as pl
from systems import MLMSystem, ClassificationSystem
import architectures


def classify(ds_path, batch_size, epochs, d_model, n_head, dim_feed, dropout, layers, token_sequence_length=1024):
    assert d_model % n_head == 0

    data_module = FMATokenDataModule(ds_path, batch_size, token_sequence_length)


    model = architectures.BERTWithEmbedding(
        d_model=d_model, n_head=n_head, dim_feed=dim_feed, dropout=dropout, layers=layers,
        max_len=token_sequence_length, output_units=16)

    mir_system = ClassificationSystem(model=model)
    trainer = pl.Trainer(
        max_epochs=epochs, progress_bar_refresh_rate=20, weights_summary='full')
    trainer.fit(mir_system, data_module)


if __name__ == '__main__':
    fire.Fire()
