from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm

from src.model import ConditionalAE, ModelConfig, count_trainable_parameters
from src.utils import (
    get_device,
    load_metadata,
    resolve_urbansound_data_dir,
    save_log_mel_spectrogram,
    unpack_audio,
)

try:
    import torchvision
except Exception:  # pragma: no cover - optional dependency
    torchvision = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency

    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_graph(self, *args, **kwargs) -> None:
            pass

        def add_scalars(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_image(self, *args, **kwargs) -> None:
            pass

        def add_hparams(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 2
    val_ratio: float = 0.1
    seed: int = 42
    pin_memory: bool = True
    grad_clip: float = 1.0
    patience: int = 15
    es_min_delta: float = 1e-4
    image_log_every: int = 10
    ckpt_path: Path = Path("checkpoints/cae_best.pt")
    log_dir: Path = Path("runs/cae")


class SpectrogramDataset(Dataset):
    """Loads saved log-mel .npy files and class labels."""

    def __init__(self, metadata: pd.DataFrame, spec_dir: str | Path, spec_t: int = 176) -> None:
        self.spec_dir = Path(spec_dir).resolve()
        self.spec_t = spec_t

        rows: list[tuple[str, int]] = []
        for _, row in metadata.iterrows():
            npy_path = self.spec_dir / f"{Path(row['slice_file_name']).stem}.npy"
            if npy_path.is_file():
                rows.append((str(npy_path), int(row["classID"])))
        self.samples = rows
        print(f"Dataset: {len(self.samples)} samples from {self.spec_dir}")

    def _pad_or_crop(self, spec: np.ndarray) -> np.ndarray:
        t = spec.shape[1]
        if t >= self.spec_t:
            return spec[:, : self.spec_t]
        pad = np.full((spec.shape[0], self.spec_t - t), spec.min(), dtype=spec.dtype)
        return np.concatenate([spec, pad], axis=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        log_mel = np.load(path)
        log_mel = self._pad_or_crop(log_mel)
        log_mel = (log_mel + 40.0) / 40.0

        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def build_dataloaders(
    metadata: pd.DataFrame,
    spec_dir: str | Path,
    config: TrainConfig,
    spec_t: int = 176,
) -> tuple[SpectrogramDataset, DataLoader, DataLoader]:
    dataset = SpectrogramDataset(metadata=metadata, spec_dir=spec_dir, spec_t=spec_t)
    if len(dataset) < 2:
        raise ValueError(
            "Need at least 2 spectrogram samples to split train/val. "
            "Run preprocessing first and verify --spec-dir."
        )

    val_size = max(1, int(config.val_ratio * len(dataset)))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return dataset, train_loader, val_loader


def build_overfit_dataloaders(
    train_loader: DataLoader,
    config: TrainConfig,
    overfit_samples: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from the same tiny subset for sanity checking."""
    base_ds = train_loader.dataset
    subset_size = min(overfit_samples, len(base_ds))
    if subset_size < 1:
        raise ValueError("Overfit subset is empty. Increase --overfit-samples.")

    indices = torch.randperm(len(base_ds), generator=torch.Generator().manual_seed(config.seed)).tolist()
    tiny_ds = Subset(base_ds, indices[:subset_size])

    tiny_batch = min(config.batch_size, subset_size)
    tiny_train_loader = DataLoader(
        tiny_ds,
        batch_size=tiny_batch,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    tiny_val_loader = DataLoader(
        tiny_ds,
        batch_size=tiny_batch,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return tiny_train_loader, tiny_val_loader


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat, x)


def _to_img(t: torch.Tensor) -> torch.Tensor:
    t = t - t.min()
    return t / (t.max() + 1e-8)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float | int | list[float] | str]:
    config.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    writer = SummaryWriter(log_dir=str(config.log_dir))

    dummy_x = torch.zeros(1, 1, 128, getattr(model, "config").spec_t, device=device)
    dummy_y = torch.zeros(1, dtype=torch.long, device=device)
    try:
        writer.add_graph(model, (dummy_x, dummy_y))
    except Exception:
        pass

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    es_counter = 0

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training"):
        model.train()
        train_running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_hat, _ = model(x, y)
            loss = reconstruction_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_running += loss.item() * x.size(0)
        train_loss = train_running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_hat, _ = model(x, y)
                val_running += reconstruction_loss(x_hat, x).item() * x.size(0)
        val_loss = val_running / len(val_loader.dataset)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        writer.add_scalar("Grad/norm", total_norm**0.5, epoch)

        if torchvision is not None and (epoch % config.image_log_every == 0 or epoch == 1):
            with torch.no_grad():
                x_sample, y_sample = next(iter(val_loader))
                x_sample = x_sample[:8].to(device)
                y_sample = y_sample[:8].to(device)
                x_hat_sample, _ = model(x_sample, y_sample)

            grid_real = torchvision.utils.make_grid(_to_img(x_sample), nrow=4)
            grid_recon = torchvision.utils.make_grid(_to_img(x_hat_sample), nrow=4)
            writer.add_image("Spectrogram/real", grid_real, epoch)
            writer.add_image("Spectrogram/reconstruction", grid_recon, epoch)

        if val_loss < best_val - config.es_min_delta:
            best_val = val_loss
            best_epoch = epoch
            es_counter = 0
            torch.save(model.state_dict(), config.ckpt_path)
        else:
            es_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{config.epochs} "
                f"train L1={train_loss:.4f} val L1={val_loss:.4f} "
                f"lr={current_lr:.2e} es={es_counter}/{config.patience}"
            )

        if es_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience}).")
            break

    writer.close()
    print(f"Best val L1: {best_val:.4f} (epoch={best_epoch}, ckpt={config.ckpt_path})")
    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "ckpt_path": str(config.ckpt_path),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean training script for Conditional AE baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("UrbanSound8K"))
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--audio-unpacked-dir", type=Path, default=None)
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--prepare-audio", action="store_true", help="Flatten fold structure into audio_unpacked.")
    parser.add_argument("--move-audio-files", action="store_true")
    parser.add_argument("--prepare-specs", action="store_true", help="Generate .npy log-mel spectrograms.")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spec-t", type=int, default=176)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=32)

    parser.add_argument("--sanity-overfit", action="store_true", help="Train/val on the same tiny subset.")
    parser.add_argument("--overfit-samples", type=int, default=32)

    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/cae_best.pt"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/cae"))
    parser.add_argument("--summary-path", type=Path, default=Path("runs/cae/train_summary.json"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _set_seed(args.seed)

    data_dir = resolve_urbansound_data_dir(args.data_dir)
    audio_dir = (args.audio_dir or (data_dir / "audio")).resolve()
    audio_unpacked_dir = (args.audio_unpacked_dir or (data_dir / "audio_unpacked")).resolve()
    spec_dir = (args.spec_dir or (data_dir / "spectrograms")).resolve()

    device = get_device()
    metadata = load_metadata(data_dir)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Spec dir: {spec_dir}")
    print(f"Metadata rows: {len(metadata)}")

    if args.prepare_audio:
        unpack_audio(audio_dir, audio_unpacked_dir, move_files=args.move_audio_files)
    if args.prepare_specs:
        save_log_mel_spectrogram(
            metadata=metadata,
            audio_dir=audio_unpacked_dir,
            output_dir=spec_dir,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        ckpt_path=args.ckpt_path,
        log_dir=args.log_dir,
    )

    _, train_loader, val_loader = build_dataloaders(
        metadata=metadata,
        spec_dir=spec_dir,
        config=train_cfg,
        spec_t=args.spec_t,
    )
    if args.sanity_overfit:
        train_loader, val_loader = build_overfit_dataloaders(
            train_loader=train_loader,
            config=train_cfg,
            overfit_samples=args.overfit_samples,
        )
        print(f"Sanity overfit mode enabled on {len(train_loader.dataset)} samples.")

    model_cfg = ModelConfig(
        n_classes=int(metadata["classID"].nunique()),
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        spec_h=args.n_mels,
        spec_t=args.spec_t,
    )
    model = ConditionalAE(config=model_cfg).to(device)
    print(f"Trainable params: {count_trainable_parameters(model):,}")

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=train_cfg,
    )

    payload = {
        "mode": "sanity_overfit" if args.sanity_overfit else "standard_train",
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "best_val": result["best_val"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": len(result["train_losses"]),
        "ckpt_path": result["ckpt_path"],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"Summary saved to {args.summary_path}")
    print(f"TensorBoard: tensorboard --logdir {train_cfg.log_dir}")


if __name__ == "__main__":
    main()

