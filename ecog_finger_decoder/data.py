import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset

# Training windows and validation split.
WINDOW   = 256 # frames
N_CHUNKS = 10
VAL_FRAC = 0.15


class WindowDataset(Dataset):
    def __init__(self, spec, glove, window=WINDOW):
        self.spec   = spec.astype('float32')
        self.glove  = glove.astype('float32')
        self.window = window


    def __len__(self):
        return self.spec.shape[-1] - self.window


    def __getitem__(self, i):
        return self.spec[..., i:i+self.window], self.glove[..., i:i+self.window]


def load_cells(paths, *keys):
    mats = [scipy.io.loadmat(path) for path in (paths if isinstance(paths, list) else [paths])]
    return [[x for mat in mats for x in _unpack(mat[key])] for key in keys]


def interleaved_split(spec, glove, n_chunks=N_CHUNKS, val_frac=VAL_FRAC):
    n_frames  = spec.shape[-1]
    chunk_len = n_frames // n_chunks

    # Validation mask, tail of every chunk.
    is_val = np.zeros(n_frames, dtype=bool)
    for i in range(n_chunks):
        start = i * chunk_len
        end   = (i + 1) * chunk_len if i < n_chunks - 1 else n_frames
        is_val[start + int((end - start) * (1 - val_frac)):end] = True

    return spec[..., ~is_val], glove[..., ~is_val], spec[..., is_val], glove[..., is_val]


def fit_scalers(spec, glove):
    ecog_scaler  = RobustScaler(unit_variance=True, quantile_range=(0.1, 0.9))
    ecog_scaler.fit(_frames_first(spec))
    glove_scaler = MinMaxScaler().fit(glove.T)
    return ecog_scaler, glove_scaler


def scale_spec(ecog_scaler, spec):
    return ecog_scaler.transform(_frames_first(spec)).T.reshape(spec.shape)


def _frames_first(spec):
    return spec.reshape(-1, spec.shape[-1]).T


def _unpack(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [x[i, 0] if x.ndim == 2 else x[i] for i in range(x.shape[0])]
    return [x]
