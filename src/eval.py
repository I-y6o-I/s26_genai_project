from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import ConditionalAE, Encoder, ModelConfig
from src.train import TrainConfig, build_dataloaders
from src.utils import (
    denorm_log_mel,
    get_device,
    load_metadata,
    resolve_urbansound_data_dir,
    spec_to_waveform,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at import time
    plt = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional at import time
    sf = None

try:
    from scipy import linalg as sla
except Exception:  # pragma: no cover - optional at import time
    sla = None


@dataclass
class EvalConfig:
    batch_size: int = 128
    noise_std: float = 0.5
    griffin_lim_iters: int = 60


def stft_mag(wave: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return spec.abs()


def mr_stft_loss(w_gt: torch.Tensor, w_hat: torch.Tensor, fft_sizes: tuple[int, ...] = (512, 1024, 2048)) -> float:
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        mag_gt = stft_mag(w_gt, n_fft, hop)
        mag_hat = stft_mag(w_hat, n_fft, hop)
        sc = (mag_gt - mag_hat).norm() / (mag_gt.norm() + 1e-8)
        lm = F.l1_loss(torch.log(mag_gt + 1e-7), torch.log(mag_hat + 1e-7))
        total += (sc + lm).item()
    return total / len(fft_sizes)


def si_sdr(target: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, target) / (np.dot(target, target) + eps)
    projection = alpha * target
    noise = estimate - projection
    return 10.0 * np.log10((projection**2).mean() / ((noise**2).mean() + eps) + eps)


class EmbeddingExtractor(nn.Module):
    """Returns latent vectors using the trained encoder."""

    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy_labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.encoder(x, dummy_labels)


@torch.no_grad()
def get_embeddings(
    specs: torch.Tensor,
    embedder: EmbeddingExtractor,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for i in range(0, len(specs), batch_size):
        batch = specs[i : i + batch_size].to(device)
        embeddings.append(embedder(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def compute_fad(emb_real: np.ndarray, emb_fake: np.ndarray) -> float:
    if sla is None:
        raise ModuleNotFoundError("scipy is required for FAD computation. Install with `pip install scipy`.")

    mu_r, sigma_r = emb_real.mean(0), np.cov(emb_real, rowvar=False)
    mu_f, sigma_f = emb_fake.mean(0), np.cov(emb_fake, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = sla.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))


def embedding_diversity(embs: np.ndarray) -> float:
    return float(embs.std(axis=0).mean())


@torch.no_grad()
def collect_reconstructions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_gt: list[torch.Tensor] = []
    all_recon: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in tqdm(val_loader, desc="Collecting reconstructions"):
        x, y = x.to(device), y.to(device)
        x_hat, _ = model(x, y)
        all_gt.append(x.cpu())
        all_recon.append(x_hat.cpu())
        all_labels.append(y.cpu())

    gt_specs = torch.cat(all_gt)
    recon_specs = torch.cat(all_recon)
    labels_val = torch.cat(all_labels)
    return gt_specs, recon_specs, labels_val


@torch.no_grad()
def generate_fake_specs(
    model: nn.Module,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
    noise_std: float = 0.5,
) -> torch.Tensor:
    fake_specs: list[torch.Tensor] = []
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i : i + batch_size].to(device)
        fake_specs.append(model.generate(batch_labels, device=device, noise_std=noise_std).cpu())
    return torch.cat(fake_specs)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: EvalConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    config = config or EvalConfig()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")

    gt_specs, recon_specs, labels_val = collect_reconstructions(model, val_loader, device)
    logmel_l1 = F.l1_loss(recon_specs, gt_specs).item()

    gt_dbs = denorm_log_mel(gt_specs)
    recon_dbs = denorm_log_mel(recon_specs)

    waves_gt: list[np.ndarray] = []
    waves_hat: list[np.ndarray] = []
    for idx in tqdm(range(len(gt_dbs)), desc="Griffin-Lim"):
        waves_gt.append(spec_to_waveform(gt_dbs[idx], n_iter=config.griffin_lim_iters))
        waves_hat.append(spec_to_waveform(recon_dbs[idx], n_iter=config.griffin_lim_iters))

    min_len = min(min(w.shape[0] for w in waves_gt), min(w.shape[0] for w in waves_hat))
    w_gt = torch.tensor(np.stack([w[:min_len] for w in waves_gt]), dtype=torch.float32)
    w_hat = torch.tensor(np.stack([w[:min_len] for w in waves_hat]), dtype=torch.float32)

    mr_stft = mr_stft_loss(w_gt, w_hat)
    si_sdr_scores = [si_sdr(w_gt[i].numpy(), w_hat[i].numpy()) for i in range(len(w_gt))]
    mean_si_sdr = float(np.mean(si_sdr_scores))

    fake_specs = generate_fake_specs(
        model,
        labels=labels_val,
        device=device,
        batch_size=config.batch_size,
        noise_std=config.noise_std,
    )
    embedder = EmbeddingExtractor(model.encoder).to(device)
    embedder.eval()
    emb_real = get_embeddings(gt_specs, embedder, device, batch_size=config.batch_size)
    emb_fake = get_embeddings(fake_specs, embedder, device, batch_size=config.batch_size)

    fad = compute_fad(emb_real, emb_fake)
    div_real = embedding_diversity(emb_real)
    div_fake = embedding_diversity(emb_fake)

    metrics = {
        "logmel_l1": logmel_l1,
        "mr_stft": mr_stft,
        "si_sdr": mean_si_sdr,
        "fad": fad,
        "diversity_real": div_real,
        "diversity_fake": div_fake,
    }
    return metrics


def save_qualitative_samples(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    count: int = 4,
    noise_std: float = 0.5,
    griffin_lim_iters: int = 60,
    sample_rate: int = 22050,
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    collected = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            x_hat, _ = model(x, y)
            x_gen = model.generate(y, device=device, noise_std=noise_std)

            x_db = denorm_log_mel(x.cpu())
            xhat_db = denorm_log_mel(x_hat.cpu())
            xgen_db = denorm_log_mel(x_gen.cpu())
            labels = y.cpu().numpy().tolist()

            for i in range(x.shape[0]):
                if collected >= count:
                    return out_dir
                sample_dir = out_dir / f"sample_{collected:03d}_class_{labels[i]}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                if plt is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                    axes[0].imshow(x_db[i], aspect="auto", origin="lower", cmap="magma")
                    axes[0].set_title("Ground Truth")
                    axes[1].imshow(xhat_db[i], aspect="auto", origin="lower", cmap="magma")
                    axes[1].set_title("Reconstruction")
                    axes[2].imshow(xgen_db[i], aspect="auto", origin="lower", cmap="magma")
                    axes[2].set_title("Generated")
                    for ax in axes:
                        ax.set_xlabel("time")
                        ax.set_ylabel("mel")
                    plt.tight_layout()
                    fig.savefig(sample_dir / "spectrograms.png", dpi=140)
                    plt.close(fig)

                if sf is not None:
                    w_gt = spec_to_waveform(x_db[i], n_iter=griffin_lim_iters)
                    w_hat = spec_to_waveform(xhat_db[i], n_iter=griffin_lim_iters)
                    w_gen = spec_to_waveform(xgen_db[i], n_iter=griffin_lim_iters)
                    sf.write(sample_dir / "gt.wav", w_gt, sample_rate)
                    sf.write(sample_dir / "recon.wav", w_hat, sample_rate)
                    sf.write(sample_dir / "gen.wav", w_gen, sample_rate)

                collected += 1
    return out_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean inference/evaluation script for Conditional AE baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("UrbanSound8K"))
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--spec-t", type=int, default=176)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=32)

    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--griffin-iters", type=int, default=60)
    parser.add_argument("--qualitative-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/inference/metrics.json"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = get_device()

    data_dir = resolve_urbansound_data_dir(args.data_dir)
    spec_dir = (args.spec_dir or (data_dir / "spectrograms")).resolve()
    metadata = load_metadata(data_dir)

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    _, _, val_loader = build_dataloaders(
        metadata=metadata,
        spec_dir=spec_dir,
        config=train_cfg,
        spec_t=args.spec_t,
    )

    model_cfg = ModelConfig(
        n_classes=int(metadata["classID"].nunique()),
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        spec_h=args.n_mels,
        spec_t=args.spec_t,
    )
    model = ConditionalAE(config=model_cfg).to(device)

    eval_cfg = EvalConfig(
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        griffin_lim_iters=args.griffin_iters,
    )
    metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        config=eval_cfg,
        checkpoint_path=args.checkpoint,
    )

    qual_dir = save_qualitative_samples(
        model=model,
        val_loader=val_loader,
        device=device,
        out_dir=args.output_dir / "qualitative",
        count=args.qualitative_count,
        noise_std=args.noise_std,
        griffin_lim_iters=args.griffin_iters,
    )

    payload = {
        "metrics": metrics,
        "checkpoint": args.checkpoint,
        "data_dir": data_dir,
        "spec_dir": spec_dir,
        "model_config": asdict(model_cfg),
        "eval_config": asdict(eval_cfg),
        "qualitative_dir": qual_dir,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(_jsonable(payload), indent=2))

    print("Quantitative metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Metrics JSON: {args.metrics_path}")
    print(f"Qualitative samples: {qual_dir}")


if __name__ == "__main__":
    main()

