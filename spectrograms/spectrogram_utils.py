import numpy as np
import torch
import librosa
import os
import pandas as pd
import ast


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def save_spec(filename, spec, meta_data={}):
    if type(spec) == np.ndarray:
        spec = torch.from_numpy(spec)
    elif type(spec) != torch.Tensor:
        raise TypeError(f"Spectrogram has to be numpy.ndarray or torch.Tensor. Is {type(spec)}.")

    meta_data.update({"data": spec})
    torch.save(meta_data, filename)


def load_spec(filename, logger=None):
    try:
        spec_dict = torch.load(filename)
        return spec_dict
    except FileNotFoundError as e:
        if logger is not None:
            logger.error(f"File not found: \n\n{e}")
    return None


def gen_spec(filename, n_fft, hop_length, sr=None, n_mels=None):
    x, sr = librosa.load(filename, sr=sr, mono=True)
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2, n_mels=n_mels)
    return mel, sr
