import torch
import torch.nn as nn


class Generator(nn.Module):
    #U-Net Generator

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # СНАЧАЛА создаём activation
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # ПОТОМ создаём блоки (они используют self.relu)
        self.enc1 = self._down_block(in_channels, features, normalize=False)
        self.enc2 = self._down_block(features, features * 2)
        self.enc3 = self._down_block(features * 2, features * 4)
        self.enc4 = self._down_block(features * 4, features * 8)
        self.enc5 = self._down_block(features * 8, features * 8)
        self.enc6 = self._down_block(features * 8, features * 8)
        self.enc7 = self._down_block(features * 8, features * 8)
        self.bottleneck = self._down_block(features * 8, features * 8, normalize=False)

        # Декодер
        self.dec1 = self._up_block(features * 8, features * 8, dropout=True)
        self.dec2 = self._up_block(features * 16, features * 8, dropout=True)
        self.dec3 = self._up_block(features * 16, features * 8, dropout=True)
        self.dec4 = self._up_block(features * 16, features * 8)
        self.dec5 = self._up_block(features * 16, features * 4)
        self.dec6 = self._up_block(features * 8, features * 2)
        self.dec7 = self._up_block(features * 4, features)

        # Выход
        self.out = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _down_block(self, in_ch, out_ch, normalize=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            self.relu  # ← Теперь self.relu уже существует!
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        return nn.Sequential(*layers)

    def _up_block(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            self.relu
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        bottleneck = self.bottleneck(e7)

        d1 = self.dec1(bottleneck)
        d1 = torch.cat([d1, e7], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)

        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)

        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)

        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        return self.out(d7)

class Discriminator(nn.Module):
   #PatchGAN Discriminator

    def __init__(self, in_channels=6):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.disc(x)