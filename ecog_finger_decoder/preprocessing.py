import numpy as np
import mne

from .config import FS_RAW, FS, N_WAVELETS, N_DELAY

# Spectral features.
F_LINE     = 60  # Hz
F_LO       = 40  # Hz
F_HI       = 300 # Hz
F_WAVELETS = np.logspace(np.log10(F_LO), np.log10(F_HI), N_WAVELETS) # Hz
F_NOTCH    = F_LINE * np.arange(1, FS_RAW // 2 // F_LINE)             # Hz


def preprocess_ecog(ecog):
    # Electrodes x samples.
    x = ecog.T

    # Per-electrode z-score and common median reference.
    x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
    x = x - np.median(x, axis=0, keepdims=True)

    # High-gamma band-pass and powerline notch.
    x = mne.filter.filter_data(x, FS_RAW, l_freq=F_LO, h_freq=F_HI, verbose='WARNING')
    x = mne.filter.notch_filter(x, FS_RAW, freqs=F_NOTCH, verbose='WARNING')

    # Morlet wavelet power at the frame rate (electrodes x wavelets x frames).
    return mne.time_frequency.tfr_array_morlet(x[None], sfreq=FS_RAW, freqs=F_WAVELETS,
                                               decim=FS_RAW // FS, output='power',
                                               n_jobs=-1, verbose='WARNING')[0]


def preprocess_glove(glove):
    return glove.T[:, ::FS_RAW // FS]


def align_delay(spec, glove):
    return spec[..., :spec.shape[-1] - N_DELAY], glove[..., N_DELAY:]
