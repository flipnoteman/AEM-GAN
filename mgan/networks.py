"""Plain convolutional encoder and generator.

Nothing memory-specific lives here.  The encoder turns an image into the
unit-norm feature the memory stores and looks up; the generator turns noise
plus one memory key into an image.  Both are sized for 28x28 inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Image [B, C, 28, 28] -> unit-norm feature [B, D].

    The output is L2-normalised because memory keys live on the unit sphere
    and similarity is cosine.  This is the discriminator's trunk.
    """

    def __init__(self, channels: int, feature_dim: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, c, 4, stride=2, padding=1),      # 28 -> 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),         # 14 -> 7
            nn.BatchNorm2d(2 * c),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),     # 7 -> 4
            nn.BatchNorm2d(4 * c),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(4 * c * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return F.normalize(self.fc(h), dim=1)


class Generator(nn.Module):
    """(noise [B, Z], key [B, D]) -> image [B, C, 28, 28] in [-1, 1].

    The key is simply concatenated to the noise; the memory decides which
    key, the generator only has to use it.
    """

    def __init__(self, latent_dim: int, feature_dim: int, channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + feature_dim, 4 * c * 7 * 7),
            nn.BatchNorm1d(4 * c * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1),   # 7 -> 14
            nn.BatchNorm2d(2 * c),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c, channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Tanh(),
        )
        self.c = c

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, key], dim=1)).view(-1, 4 * self.c, 7, 7)
        return self.deconv(h)
