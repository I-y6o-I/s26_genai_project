from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int = 10
    embed_dim: int = 32
    latent_dim: int = 128
    base_ch: int = 32
    spec_h: int = 128
    spec_t: int = 176


class ResBlock(nn.Module):
    """Residual block with optional spatial scaling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, transpose: bool = False) -> None:
        super().__init__()
        conv = nn.ConvTranspose2d if transpose else nn.Conv2d

        if transpose:
            self.conv1 = conv(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=stride - 1,
            )
        else:
            self.conv1 = conv(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip: nn.Module | None = None
        if in_ch != out_ch or stride != 1:
            if transpose:
                self.skip = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=1,
                        stride=stride,
                        output_padding=stride - 1,
                    ),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_ch),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_classes, config.embed_dim)

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.blocks = nn.Sequential(
            ResBlock(1, ch[0], stride=2),
            ResBlock(ch[0], ch[1], stride=2),
            ResBlock(ch[1], ch[2], stride=2),
            ResBlock(ch[2], ch[3], stride=2),
        )

        h_out = config.spec_h // 16
        t_out = config.spec_t // 16
        flat_dim = ch[3] * h_out * t_out

        self.embed_proj = nn.Linear(config.embed_dim, ch[3])
        self.fc = nn.Linear(flat_dim + ch[3], config.latent_dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self.blocks(x)
        flat = feat.flatten(1)
        emb = self.embedding(labels)
        emb = self.embed_proj(emb)
        return self.fc(torch.cat([flat, emb], dim=1))


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_classes, config.embed_dim)

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.h_out = config.spec_h // 16
        self.t_out = config.spec_t // 16
        self.start_ch = ch[3]

        self.fc = nn.Linear(config.latent_dim + config.embed_dim, ch[3] * self.h_out * self.t_out)
        self.blocks = nn.Sequential(
            ResBlock(ch[3], ch[2], stride=2, transpose=True),
            ResBlock(ch[2], ch[1], stride=2, transpose=True),
            ResBlock(ch[1], ch[0], stride=2, transpose=True),
            ResBlock(ch[0], ch[0], stride=2, transpose=True),
        )
        self.head = nn.Sequential(nn.Conv2d(ch[0], 1, kernel_size=1), nn.Tanh())

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(labels)
        x = self.fc(torch.cat([z, emb], dim=1))
        x = x.view(-1, self.start_ch, self.h_out, self.t_out)
        x = self.blocks(x)
        return self.head(x)


class ConditionalAE(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, labels)
        x_hat = self.decoder(z, labels)
        return x_hat, z

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        device: torch.device | None = None,
        noise_std: float = 0.5,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)
        z = torch.randn(labels.shape[0], self.config.latent_dim, device=device) * noise_std
        return self.decoder(z, labels)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

