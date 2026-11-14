import json
import pickle

import numpy as np
import scipy.io
import torch

from ecog_finger_decoder.config import (CACHE_DIR, CKPT_DIR, OUT_MAT, TEST_MAT, TEST_KEY,
                                        N_ELECTRODES, SEEDS)
from ecog_finger_decoder.data import load_cells, scale_spec
from ecog_finger_decoder.inference import predict_spec, shift_delay, upsample
from ecog_finger_decoder.model import UNet1D
from ecog_finger_decoder.training import FingerModule


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ecogs_test = load_cells(TEST_MAT, TEST_KEY)[0]
    with open(CACHE_DIR / 'scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open(CKPT_DIR / 'runs.json') as f:
        runs = json.load(f)

    predicted_dg = np.empty((len(N_ELECTRODES), 1), dtype=object)
    for i, n_electrodes in enumerate(N_ELECTRODES):
        print(f'Subject {i+1}')
        ecog_scaler, glove_scaler = scalers[i]
        spec = scale_spec(ecog_scaler, np.load(CACHE_DIR / f'test_subj{i+1}.npz')['spec'])

        # Seed ensemble.
        glove_hats = []
        for seed in SEEDS:
            run = runs.get(str(seed), {}).get(str(i))
            if run is None:
                print(f'  Seed {seed} missing, skipped')
                continue
            module = FingerModule.load_from_checkpoint(run['ckpt'], model=UNet1D(n_electrodes))
            glove_hats.append(predict_spec(module.model.to(device).eval(), spec, device))
        glove_hat = np.mean(glove_hats, axis=0)

        # Delay compensation, unscaling and upsampling.
        glove_hat_raw = glove_scaler.inverse_transform(shift_delay(glove_hat).T).T
        predicted_dg[i, 0] = upsample(glove_hat_raw, ecogs_test[i].shape[0])
        print(f'  {len(glove_hats)} seeds, output {predicted_dg[i, 0].shape}')

    scipy.io.savemat(OUT_MAT, {'predicted_dg': predicted_dg})
    print(f'Saved to {OUT_MAT}')


if __name__ == '__main__':
    main()
