from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int = 10
    embed_dim: int = 32
    latent_dim: int = 128
    base_ch: int = 32
    spec_h: int = 128
    spec_t: int = 176
    vq_num_embeddings: int = 512
    vq_commitment_beta: float = 0.25
    vq_ema_decay: float = 0.99
    vq_ema_eps: float = 1e-5


class ResBlock(nn.Module):
    """Residual block with optional spatial scaling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int | tuple[int, int] = 1,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(stride, int):
            stride_2d = (stride, stride)
        else:
            stride_2d = stride
        output_padding: tuple[int, int] = (max(stride_2d[0] - 1, 0), max(stride_2d[1] - 1, 0))
        if transpose:
            self.conv1 = nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride_2d,
                padding=1,
                output_padding=output_padding,
            )
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride_2d, padding=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip: nn.Module | None = None
        if in_ch != out_ch or stride_2d != (1, 1):
            if transpose:
                self.skip = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=1,
                        stride=stride_2d,
                        output_padding=output_padding,
                    ),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride_2d),
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
            ResBlock(1, ch[0], stride=(2, 2)),
            ResBlock(ch[0], ch[1], stride=(2, 2)),
            ResBlock(ch[1], ch[2], stride=(1, 2)),
            ResBlock(ch[2], ch[3], stride=1),
        )

        h_out = config.spec_h // 4
        t_out = config.spec_t // 8
        flat_dim = ch[3] * h_out * t_out

        self.embed_proj = nn.Linear(config.embed_dim, ch[3])
        self.fc = nn.Linear(flat_dim + ch[3], config.latent_dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self.blocks(x)
        flat = feat.flatten(1)
        emb = self.embedding(labels)
        emb = self.embed_proj(emb)
        return self.fc(torch.cat([flat, emb], dim=1))


class VariationalEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_classes, config.embed_dim)

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.blocks = nn.Sequential(
            ResBlock(1, ch[0], stride=(2, 2)),
            ResBlock(ch[0], ch[1], stride=(2, 2)),
            ResBlock(ch[1], ch[2], stride=(1, 2)),
            ResBlock(ch[2], ch[3], stride=1),
        )

        h_out = config.spec_h // 4
        t_out = config.spec_t // 8
        flat_dim = ch[3] * h_out * t_out

        self.embed_proj = nn.Linear(config.embed_dim, ch[3])
        self.fc_mu = nn.Linear(flat_dim + ch[3], config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + ch[3], config.latent_dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.blocks(x)
        flat = feat.flatten(1)
        emb = self.embedding(labels)
        emb = self.embed_proj(emb)
        joined = torch.cat([flat, emb], dim=1)
        return self.fc_mu(joined), self.fc_logvar(joined)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_classes, config.embed_dim)

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.h_out = config.spec_h // 4
        self.t_out = config.spec_t // 8
        self.start_ch = ch[3]

        self.fc = nn.Linear(config.latent_dim + config.embed_dim, ch[3] * self.h_out * self.t_out)
        self.blocks = nn.Sequential(
            ResBlock(ch[3], ch[3], stride=1, transpose=True),
            ResBlock(ch[3], ch[2], stride=(1, 2), transpose=True),
            ResBlock(ch[2], ch[1], stride=(2, 2), transpose=True),
            ResBlock(ch[1], ch[0], stride=(2, 2), transpose=True),
        )
        self.head = nn.Sequential(nn.Conv2d(ch[0], 1, kernel_size=1), nn.Tanh())

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(labels)
        x = self.fc(torch.cat([z, emb], dim=1))
        x = x.view(-1, self.start_ch, self.h_out, self.t_out)
        x = self.blocks(x)
        return self.head(x)


class SpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.blocks = nn.Sequential(
            ResBlock(1, ch[0], stride=(2, 2)),
            ResBlock(ch[0], ch[1], stride=(2, 2)),
            ResBlock(ch[1], ch[2], stride=(1, 2)),
            ResBlock(ch[2], ch[3], stride=1),
        )
        self.to_latent = nn.Conv2d(ch[3], config.latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        feat = self.blocks(x)
        return self.to_latent(feat)


class SpatialDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_classes, config.embed_dim)

        ch = [config.base_ch * (2**i) for i in range(4)]
        self.h_out = config.spec_h // 4
        self.t_out = config.spec_t // 8
        self.in_proj = nn.Conv2d(config.latent_dim, ch[3], kernel_size=1)
        self.embed_proj = nn.Linear(config.embed_dim, ch[3])

        self.blocks = nn.Sequential(
            ResBlock(ch[3], ch[3], stride=1, transpose=True),
            ResBlock(ch[3], ch[2], stride=(1, 2), transpose=True),
            ResBlock(ch[2], ch[1], stride=(2, 2), transpose=True),
            ResBlock(ch[1], ch[0], stride=(2, 2), transpose=True),
        )
        self.head = nn.Sequential(nn.Conv2d(ch[0], 1, kernel_size=1), nn.Tanh())

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        emb = self.embed_proj(self.embedding(labels))
        x = x + emb[:, :, None, None]
        x = self.blocks(x)
        return self.head(x)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float,
        ema_decay: float,
        ema_eps: float,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.requires_grad = False
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_e.ndim != 4:
            raise ValueError(f"Expected z_e with shape (B,D,H,W), got {tuple(z_e.shape)}")
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, self.embedding_dim)
        codebook = self.embedding.weight

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * flat @ codebook.t()
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(encoding_indices, num_classes=self.num_embeddings).type(flat.dtype)
        z_q_flat = encodings @ codebook
        z_q = z_q_flat.view_as(z_e_perm).permute(0, 3, 1, 2).contiguous()

        if self.training:
            with torch.no_grad():
                ema_cluster_size = cast(torch.Tensor, self.ema_cluster_size)
                ema_embedding = cast(torch.Tensor, self.ema_embedding)

                encodings_sum = encodings.sum(dim=0)
                embedding_sum = encodings.t() @ flat

                ema_cluster_size.mul_(self.ema_decay).add_(encodings_sum, alpha=1.0 - self.ema_decay)
                ema_embedding.mul_(self.ema_decay).add_(embedding_sum, alpha=1.0 - self.ema_decay)

                n = ema_cluster_size.sum()
                cluster_size = (
                    (ema_cluster_size + self.ema_eps)
                    / (n + self.num_embeddings * self.ema_eps)
                    * n
                )
                normalized_embedding = ema_embedding / cluster_size.unsqueeze(1).clamp_min(self.ema_eps)
                self.embedding.weight.data.copy_(normalized_embedding)

        commitment_loss = torch.mean((z_e - z_q.detach()) ** 2)
        vq_loss = self.commitment_beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        b, _, h, w = z_e.shape
        return z_q_st, {
            "z_e": z_e,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "encoding_indices": encoding_indices.view(b, h, w),
        }



@dataclass(frozen=True)
class PriorConfig:
    num_embeddings: int = 512
    n_classes: int = 10
    hidden_dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 4096
    max_h_tokens: int = 64
    max_w_tokens: int = 64
    top_k: int = 50


class TransformerPrior(nn.Module):
    def __init__(self, config: PriorConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_token_id = config.num_embeddings

        self.token_embedding = nn.Embedding(config.num_embeddings, config.hidden_dim)
        self.bos_embedding = nn.Parameter(torch.zeros(config.hidden_dim))
        self.pos_emb_h = nn.Embedding(config.max_h_tokens, config.hidden_dim)
        self.pos_emb_w = nn.Embedding(config.max_w_tokens, config.hidden_dim)
        self.class_embedding = nn.Embedding(config.n_classes, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_embeddings, bias=False)
        self.head.weight = self.token_embedding.weight

    def _flatten_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if indices.ndim == 3:
            b, h, w = indices.shape
            return indices.permute(0, 2, 1).contiguous().view(b, w * h), (h, w)
        if indices.ndim == 2:
            return indices, None
        raise ValueError(f"Expected indices of shape (B,H,W) or (B,T), got {tuple(indices.shape)}")

    def _unflatten_indices(self, flat: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t = flat.shape
        if t != h * w:
            raise ValueError(f"Cannot unflatten sequence length {t} into (H,W)=({h},{w}).")
        return flat.view(b, w, h).permute(0, 2, 1).contiguous()

    def _build_shifted_input(self, target: torch.Tensor) -> torch.Tensor:
        b, t = target.shape
        x = torch.empty((b, t), dtype=torch.long, device=target.device)
        x[:, 0] = self.bos_token_id
        if t > 1:
            x[:, 1:] = target[:, :-1]
        return x

    def _build_2d_positions(self, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if h > self.config.max_h_tokens:
            raise ValueError(f"H={h} exceeds max_h_tokens={self.config.max_h_tokens}.")
        if w > self.config.max_w_tokens:
            raise ValueError(f"W={w} exceeds max_w_tokens={self.config.max_w_tokens}.")
        h_ids = torch.arange(h, device=device).repeat(w)
        w_ids = torch.arange(w, device=device).repeat_interleave(h)
        return h_ids, w_ids

    def _embed_input_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        clipped = input_tokens.clamp(max=self.config.num_embeddings - 1)
        token_emb = self.token_embedding(clipped)
        bos_mask = input_tokens.eq(self.bos_token_id).unsqueeze(-1)
        return torch.where(bos_mask, self.bos_embedding.view(1, 1, -1), token_emb)

    def _forward_input_tokens(
        self,
        input_tokens: torch.Tensor,
        labels: torch.Tensor,
        spatial_shape: tuple[int, int] | None,
    ) -> torch.Tensor:
        b, t = input_tokens.shape
        if t > self.config.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.config.max_seq_len}.")

        token_emb = self._embed_input_tokens(input_tokens)
        if spatial_shape is not None:
            h, w = spatial_shape
            if h * w != t:
                raise ValueError(f"Spatial shape {(h, w)} does not match sequence length {t}.")
            h_ids, w_ids = self._build_2d_positions(h, w, input_tokens.device)
            pos_emb = self.pos_emb_h(h_ids).unsqueeze(0) + self.pos_emb_w(w_ids).unsqueeze(0)
            pos_emb = pos_emb.expand(b, -1, -1)
        else:
            if t > self.config.max_h_tokens * self.config.max_w_tokens:
                raise ValueError(
                    "2D positional embedding requires spatial shape for long flat sequences. "
                    f"Got t={t}, max supported without shape={self.config.max_h_tokens * self.config.max_w_tokens}."
                )
            h_ids = torch.zeros((t,), dtype=torch.long, device=input_tokens.device)
            w_ids = torch.arange(t, device=input_tokens.device)
            w_ids = w_ids.clamp(max=self.config.max_w_tokens - 1)
            pos_emb = self.pos_emb_h(h_ids).unsqueeze(0) + self.pos_emb_w(w_ids).unsqueeze(0)
            pos_emb = pos_emb.expand(b, -1, -1)

        class_emb = self.class_embedding(labels.long())

        hidden = token_emb + pos_emb
        hidden[:, 0, :] = hidden[:, 0, :] + class_emb
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(torch.ones((t, t), device=input_tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.ln_f(hidden)
        return self.head(hidden)

    def forward(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target, spatial_shape = self._flatten_indices(indices.long())
        input_tokens = self._build_shifted_input(target)
        return self._forward_input_tokens(input_tokens, labels, spatial_shape=spatial_shape)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        labels = labels.to(device).long()
        b = labels.shape[0]
        t = h * w
        if t > self.config.max_seq_len:
            raise ValueError(f"Requested sequence length {t} exceeds max_seq_len={self.config.max_seq_len}.")

        shifted_input = torch.full((b, t), self.bos_token_id, dtype=torch.long, device=device)
        generated = torch.zeros((b, t), dtype=torch.long, device=device)
        temp = max(float(temperature), 1e-6)
        k = top_k if top_k is not None else self.config.top_k
        k = max(1, min(int(k), self.config.num_embeddings))

        for pos in range(t):
            logits = self._forward_input_tokens(shifted_input, labels, spatial_shape=(h, w))[:, pos, :] / temp
            top_values, top_indices = torch.topk(logits, k=k, dim=-1)
            top_probs = torch.softmax(top_values, dim=-1)
            sampled_local = torch.multinomial(top_probs, num_samples=1)
            token = top_indices.gather(1, sampled_local).squeeze(1)
            generated[:, pos] = token
            if pos + 1 < t:
                shifted_input[:, pos + 1] = token

        return self._unflatten_indices(generated, h=h, w=w)


def build_prior(config: PriorConfig) -> TransformerPrior:
    return TransformerPrior(config)


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


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ConditionalVAE(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = VariationalEncoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu, logvar = self.encoder(x, labels)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z, labels)
        return x_hat, {"z": z, "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        device: torch.device | None = None,
        noise_std: float = 1.0,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)
        z = torch.randn(labels.shape[0], self.config.latent_dim, device=device) * noise_std
        return self.decoder(z, labels)


class ConditionalVQVAE(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = SpatialEncoder(self.config)
        self.quantizer = VectorQuantizer(
            num_embeddings=self.config.vq_num_embeddings,
            embedding_dim=self.config.latent_dim,
            commitment_beta=self.config.vq_commitment_beta,
            ema_decay=self.config.vq_ema_decay,
            ema_eps=self.config.vq_ema_eps,
        )
        self.decoder = SpatialDecoder(self.config)
        self.h_out = self.config.spec_h // 4
        self.t_out = self.config.spec_t // 8

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_e = self.encoder(x, labels)
        z_q, q_stats = self.quantizer(z_e)
        x_hat = self.decoder(z_q, labels)
        q_stats["z"] = z_q
        return x_hat, q_stats

    @torch.no_grad()
    def encode_code_indices(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x, labels)
        _, q_stats = self.quantizer(z_e)
        return q_stats["encoding_indices"].long()

    @torch.no_grad()
    def decode_code_indices(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = self.quantizer.embedding(indices.long())
        z = z.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z, labels)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        device: torch.device | None = None,
        noise_std: float = 1.0,
        prior: nn.Module | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        allow_random_fallback: bool = False,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)

        if prior is not None:
            sample_fn = getattr(prior, "sample", None)
            if sample_fn is None or not callable(sample_fn):
                raise ValueError("Provided prior must implement sample(labels, h, w, device, temperature).")
            indices = cast(
                torch.Tensor,
                sample_fn(
                    labels=labels,
                    h=self.h_out,
                    w=self.t_out,
                    device=device,
                    temperature=temperature,
                    top_k=top_k,
                ),
            )
            return self.decode_code_indices(indices, labels)

        if not allow_random_fallback:
            raise RuntimeError(
                "VQ-VAE generation requires a trained prior. "
                "Pass prior=... or set allow_random_fallback=True for debug-only random generation."
            )

        indices = torch.randint(
            low=0,
            high=self.quantizer.num_embeddings,
            size=(labels.shape[0], self.h_out, self.t_out),
            device=device,
        )
        x_hat = self.decode_code_indices(indices, labels)
        if noise_std > 0:
            x_hat = x_hat + torch.randn_like(x_hat) * (0.02 * noise_std)
        return x_hat.clamp(-1.0, 1.0)


def unpack_model_output(
    output: tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x_hat, latent = output
    if isinstance(latent, dict):
        z = latent["z"]
        mu = latent.get("mu")
        logvar = latent.get("logvar")
        return x_hat, z, mu, logvar
    return x_hat, latent, None, None


def unpack_aux_losses(output: tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    _, latent = output
    if isinstance(latent, dict):
        return {k: v for k, v in latent.items() if isinstance(v, torch.Tensor)}
    return {}


def build_model(
    model_type: Literal["ae", "vae", "vqvae"],
    config: ModelConfig,
) -> nn.Module:
    if model_type == "ae":
        return ConditionalAE(config=config)
    if model_type == "vae":
        return ConditionalVAE(config=config)
    if model_type == "vqvae":
        return ConditionalVQVAE(config=config)

    raise ValueError(f"Unsupported model_type={model_type}. Use 'ae', 'vae', or 'vqvae'.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

