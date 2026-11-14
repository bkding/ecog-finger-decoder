import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch

from .config import N_DELAY, SIGMA
from .model import pad_len


def predict_spec(model, spec, device):
    n_frames = spec.shape[-1]
    spec_pad = np.pad(spec, ((0, 0), (0, 0), (0, pad_len(n_frames))), mode='reflect')

    # Forward pass, cropped to the input length.
    with torch.no_grad():
        x = torch.from_numpy(spec_pad.astype('float32'))[None].to(device)
        glove_hat = model(x)[0].cpu().numpy()[:, :n_frames]

    return scipy.ndimage.gaussian_filter1d(glove_hat, sigma=SIGMA, axis=1)


def shift_delay(glove_hat):
    glove_hat_shift = np.empty_like(glove_hat)
    glove_hat_shift[:, N_DELAY:] = glove_hat[:, :-N_DELAY]
    glove_hat_shift[:, :N_DELAY] = glove_hat[:, :1]
    return glove_hat_shift


def upsample(glove_hat, n_samples):
    # Natural cubic spline to the raw sample rate (samples x fingers).
    t_frames  = np.linspace(0, n_samples - 1, glove_hat.shape[1])
    t_samples = np.arange(n_samples)
    spline = scipy.interpolate.CubicSpline(t_frames, glove_hat, axis=1, bc_type='natural')
    glove_hat_up = np.ascontiguousarray(spline(t_samples).T)

    # Endpoint hold.
    glove_hat_up[0]  = glove_hat[:, 0]
    glove_hat_up[-1] = glove_hat[:, -1]
    return glove_hat_up
