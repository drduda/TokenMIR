import os
import numpy as np
import logging

from multiprocessing import Pool

from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio import FilteredSpectrogramProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.filters import MelFilterbank
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.io.audio import LoadAudioFileError


def signal(audiofile, sample_rate=22050, num_channels=1):
    try:
        signal_processor = SignalProcessor(sample_rate=sample_rate, num_channels=num_channels)
        return signal_processor(audiofile)
    except LoadAudioFileError:
        raise RuntimeError(f"Could not load audio file '{audiofile}'.")


def melspec_comp(s, window_size=1024, hop_size=512, num_bands=130, timestamp=False):
    """
    :param s: Madmom audio signal
    :param window_size:
    :param hop_size:
    :param num_bands:
    :param timestamp: Whether to add a timestamp to the output spectrogram
    :return:
    """

    # init audio processors
    framed_signal_processor = FramedSignalProcessor(frame_size=window_size, hop_size=hop_size)
    stft_processor = ShortTimeFourierTransformProcessor()
    filtspec_processor = FilteredSpectrogramProcessor(filterbank=MelFilterbank, fmin=30.0, fmax=s.sample_rate / 2,
                                                      num_bands=num_bands)

    # init audio processing chain
    processing_chain = SequentialProcessor([framed_signal_processor, stft_processor, filtspec_processor])

    # extract Mel spectrogram
    X = processing_chain(s)

    # magnitude compression like Park et al.
    X = np.log(10 * np.abs(X) + 1)

    # timestamps
    if timestamp:
        duration = len(s) / s.sample_rate
        frame_len = duration / X.shape[0]
        timestamps = np.linspace(frame_len, duration, X.shape[0]).reshape(-1, 1)
        X = np.concatenate((timestamps, X), axis=1)

    return X


def batch_extract(files, destination_folder, n_processes=None):
    """
    :param files: Paths to audio (mp3) files
    :param destination_folder: Folder where to store the resulting Mel spectrograms
    :param n_processes: Number of processes. If None then the number returned by os.cpu_count() is used.
    :return:
    """
    destination_files = [os.path.join(destination_folder, os.path.basename(file).split('.')[-2]) for file in files]
    args = zip(files, destination_files)
    with Pool(n_processes) as p:
        p.starmap(extract, args)


def extract(source_file, destination_file):
    try:
        s = signal(source_file)
        mel_spectrogram = melspec_comp(s)
        np.save(destination_file, mel_spectrogram)
        return mel_spectrogram
    except Exception as e:
        logging.error(f"An error occurred while extracting '{source_file}': {str(e)}")
    print(f'Completed source={source_file}, destination={destination_file}.')
    return None
