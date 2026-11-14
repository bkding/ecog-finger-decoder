import json
import pickle
import random
import time

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from ecog_finger_decoder.config import CACHE_DIR, CKPT_DIR, N_ELECTRODES, SEEDS, MAX_EPOCHS
from ecog_finger_decoder.data import WindowDataset, fit_scalers, scale_spec, interleaved_split
from ecog_finger_decoder.model import UNet1D
from ecog_finger_decoder.training import FingerModule, PearsonCallback

# Loaders and early stopping.
BATCH_SIZE = 128
PATIENCE   = 12 # epochs
MIN_DELTA  = 0.003


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Scaled training recordings.
    specs, gloves, scalers = [], [], []
    for i in range(len(N_ELECTRODES)):
        cache = np.load(CACHE_DIR / f'train_subj{i+1}.npz')
        ecog_scaler, glove_scaler = fit_scalers(cache['spec'], cache['glove'])
        specs.append(scale_spec(ecog_scaler, cache['spec']))
        gloves.append(glove_scaler.transform(cache['glove'].T).T)
        scalers.append((ecog_scaler, glove_scaler))
    with open(CACHE_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Training runs.
    runs = {}
    t_start = time.time()
    for seed in SEEDS:
        set_seed(seed)
        runs[seed] = {}
        for i, n_electrodes in enumerate(N_ELECTRODES):
            print(f'\n--- Seed {seed}, Subject {i+1} ---')
            runs[seed][i] = fit_subject(specs[i], gloves[i], n_electrodes,
                                        f'seed{seed}_subj{i+1}')
            print(f'  Best val r: {runs[seed][i]["r"]:.4f} | '
                  f'Total: {(time.time() - t_start) / 60:.1f} min')

    with open(CKPT_DIR / 'runs.json', 'w') as f:
        json.dump(runs, f, indent=2)

    # Summary.
    r_runs = np.array([[runs[seed][i]['r'] for i in range(len(N_ELECTRODES))] for seed in SEEDS])
    for i, r in enumerate(r_runs.mean(axis=0)):
        print(f'Subject {i+1}: mean val r {r:.4f}')
    print(f'Overall: mean val r {r_runs.mean():.4f}')


def fit_subject(spec, glove, n_electrodes, name):
    # Interleaved split and window loaders.
    spec_train, glove_train, spec_val, glove_val = interleaved_split(spec, glove)
    print(f'  Train: {spec_train.shape[-1]} frames, Val: {spec_val.shape[-1]} frames')
    loader_train = DataLoader(WindowDataset(spec_train, glove_train), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    loader_val   = DataLoader(WindowDataset(spec_val, glove_val), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model and callbacks.
    module = FingerModule(UNet1D(n_electrodes))
    stop = EarlyStopping(monitor='val_r', mode='max', patience=PATIENCE,
                         min_delta=MIN_DELTA, verbose=True)
    ckpt = ModelCheckpoint(dirpath=CKPT_DIR, filename=name + '-{epoch:02d}-{val_r:.4f}',
                           monitor='val_r', mode='max', save_top_k=1)

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1,
                         callbacks=[PearsonCallback(spec_val, glove_val), ckpt, stop],
                         logger=False,
                         enable_progress_bar=True)
    trainer.fit(module, loader_train, loader_val)

    r_best = ckpt.best_model_score.item() if ckpt.best_model_score else 0
    return {'ckpt': ckpt.best_model_path, 'r': r_best}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    main()
