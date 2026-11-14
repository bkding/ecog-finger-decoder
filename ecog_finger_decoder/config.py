from pathlib import Path

# Paths.
DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache'
CKPT_DIR  = Path('checkpoints')
OUT_MAT   = Path('predicted_dg.mat')

# Raw recordings.
TRAIN_MAT  = DATA_DIR / 'raw_training_data.mat'
TRAIN_KEYS = ('train_ecog', 'train_dg')
TEST_MAT   = DATA_DIR / 'leaderboard_data.mat'
TEST_KEY   = 'leaderboard_ecog'

# BCI Competition IV dataset 4, one file per subject.
# TRAIN_MAT  = [DATA_DIR / f'sub{i}_comp.mat' for i in (1, 2, 3)]
# TRAIN_KEYS = ('train_data', 'train_dg')
# TEST_MAT   = TRAIN_MAT
# TEST_KEY   = 'test_data'

# Recording.
FS_RAW       = 1000         # Hz
N_ELECTRODES = (62, 48, 64) # per subject
N_FINGERS    = 5

# Features.
FS         = 100             # Hz, feature and target rate
N_WAVELETS = 40
DELAY      = 0.2             # s, ECoG leads finger motion
N_DELAY    = int(DELAY * FS) # frames

# Training runs.
SEEDS      = (42, 123, 456)
MAX_EPOCHS = 50

# Postprocessing.
SIGMA = 6 # frames, Gaussian smoothing
