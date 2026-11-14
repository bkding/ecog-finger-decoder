import torch
import torch.nn as nn

from .config import N_WAVELETS, N_FINGERS

# Network.
WIDTHS     = (32, 32, 64, 64, 128, 128) # channels per encoder level
KERNELS    = (7, 7, 5, 5, 5)
P_DROP     = 0.1
T_MULTIPLE = 2 ** len(KERNELS)          # frames, input length divisor


class UNet1D(nn.Module):
    def __init__(self, n_electrodes, n_wavelets=N_WAVELETS, n_out=N_FINGERS):
        super().__init__()
        depth = len(KERNELS)

        # Electrode-wavelet mixing.
        self.mix = conv_block(n_electrodes * n_wavelets, WIDTHS[0], 3)

        # Encoder.
        self.encoder = nn.ModuleList([conv_block(WIDTHS[i], WIDTHS[i+1], KERNELS[i], pool=2)
                                      for i in range(depth)])

        # Decoder, deepest level first.
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth)):
            c_in = WIDTHS[i+1] if i == depth - 1 else 2 * WIDTHS[i+1]
            self.decoder.append(nn.Sequential(conv_block(c_in, WIDTHS[i], KERNELS[i]),
                                              nn.Upsample(scale_factor=2, mode='linear',
                                                          align_corners=False)))

        # Per-finger readout.
        self.head = nn.Conv1d(2 * WIDTHS[0], n_out, 1, padding='same')


    def forward(self, x):
        # Electrode and wavelet axes as input channels.
        x = self.mix(x.flatten(1, 2))

        # Encoder with stored skips.
        skips = []
        for block in self.encoder:
            skips.append(x)
            x = block(x)

        # Decoder with skip concatenation.
        for block, skip in zip(self.decoder, reversed(skips)):
            x = torch.cat((block(x), skip), dim=1)
        return self.head(x)


class ChannelNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(-2, -1)).transpose(-2, -1)


def conv_block(c_in, c_out, kernel, pool=1):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel, padding='same', bias=False),
                         ChannelNorm(c_out),
                         nn.GELU(),
                         nn.Dropout(P_DROP),
                         nn.MaxPool1d(pool, pool))


def pad_len(n_frames):
    return (T_MULTIPLE - n_frames % T_MULTIPLE) % T_MULTIPLE
