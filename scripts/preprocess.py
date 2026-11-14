import numpy as np

from ecog_finger_decoder.config import CACHE_DIR, TRAIN_MAT, TRAIN_KEYS, TEST_MAT, TEST_KEY
from ecog_finger_decoder.data import load_cells
from ecog_finger_decoder.preprocessing import preprocess_ecog, preprocess_glove, align_delay


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ecogs, gloves = load_cells(TRAIN_MAT, *TRAIN_KEYS)
    ecogs_test    = load_cells(TEST_MAT, TEST_KEY)[0]

    for i, (ecog, glove, ecog_test) in enumerate(zip(ecogs, gloves, ecogs_test)):
        print(f'Subject {i+1}')

        # Delay-aligned training pair.
        spec, glove = align_delay(preprocess_ecog(ecog), preprocess_glove(glove))
        np.savez(CACHE_DIR / f'train_subj{i+1}.npz', spec=spec, glove=glove)

        # Unlabeled test features.
        spec_test = preprocess_ecog(ecog_test)
        np.savez(CACHE_DIR / f'test_subj{i+1}.npz', spec=spec_test)
        print(f'  train {spec.shape}, glove {glove.shape}, test {spec_test.shape}')


if __name__ == '__main__':
    main()
