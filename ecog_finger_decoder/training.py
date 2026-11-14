import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .config import MAX_EPOCHS, N_FINGERS, SIGMA
from .model import pad_len

# Optimizer.
LR           = 8.42e-5
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-6


class FingerModule(pl.LightningModule):
    def __init__(self, model, max_epochs=MAX_EPOCHS):
        super().__init__()
        self.model      = model
        self.max_epochs = max_epochs


    def training_step(self, batch, batch_idx):
        mse, cos = self._losses(batch, 'train')
        return 0.5 * mse + 0.5 * (1.0 - cos)


    def validation_step(self, batch, batch_idx):
        self._losses(batch, 'val')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs,
                                                               eta_min=LR_MIN)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def _losses(self, batch, stage):
        x, y = batch
        y_hat = self.model(x)
        mse = F.mse_loss(y_hat, y)
        cos = F.cosine_similarity(y_hat, y, dim=-1, eps=1e-8).mean()
        self.log(f'{stage}_mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log(f'{stage}_cos', cos, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return mse, cos


class PearsonCallback(Callback):
    def __init__(self, spec, glove):
        super().__init__()
        self.spec  = spec
        self.glove = glove


    def on_validation_epoch_end(self, trainer, pl_module):
        # Full-recording forward pass.
        n_frames = self.spec.shape[-1]
        with torch.no_grad():
            x = torch.from_numpy(self.spec.astype('float32')).to(pl_module.device)
            x = F.pad(x[None], (0, pad_len(n_frames), 0, 0), mode='replicate')
            glove_hat = pl_module.model(x)[0, :, :n_frames].cpu().numpy()

        # Smoothed per-finger correlation.
        glove_hat = scipy.ndimage.gaussian_filter1d(glove_hat, sigma=SIGMA, axis=1)
        r = [pearson(glove_hat[f], self.glove[f]) for f in range(N_FINGERS)]
        pl_module.log('val_r', np.mean(r), on_step=False, on_epoch=True, prog_bar=True,
                      logger=False)
        print(f'    val r per finger: {np.round(r, 3)}, mean {np.mean(r):.4f}')


def pearson(x, y):
    r = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(r) else r
