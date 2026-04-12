from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import ConditionalVQVAE, ModelConfig, build_model
from src.model import PriorConfig, build_prior
from src.train import TrainConfig, build_dataloaders
from src.utils import get_device, load_metadata, make_run_name, make_run_paths, resolve_urbansound_data_dir


@dataclass
class PriorTrainConfig:
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 15
    warmup_start_factor: float = 1
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    batch_size: int = 64
    num_workers: int = 2
    val_ratio: float = 0.1
    seed: int = 42
    patience: int = 8
    es_min_delta: float = 1e-4
    ckpt_path: Path = Path("checkpoints/prior_best.pt")
    summary_path: Path = Path("prior_summary.json")


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


@torch.no_grad()
def extract_indices(vqvae: ConditionalVQVAE, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return vqvae.encode_code_indices(x, y)


def train_prior(
    prior: torch.nn.Module,
    vqvae: ConditionalVQVAE,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: PriorTrainConfig,
) -> tuple[float, int]:
    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(prior.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    warmup_steps = max(0, min(cfg.warmup_epochs * steps_per_epoch, total_steps - 1))
    start_factor = max(float(cfg.warmup_start_factor), 1e-8)
    min_lr_factor = max(min(float(cfg.min_lr) / max(float(cfg.lr), 1e-12), 1.0), 0.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            alpha = float(step + 1) / float(warmup_steps)
            return start_factor + alpha * (1.0 - start_factor)

        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    best_epoch = 0
    es_counter = 0

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Prior training"):
        prior.train()
        train_running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                indices = extract_indices(vqvae, x, y)

            logits = prior(indices, y)
            targets = indices.view(indices.shape[0], -1).long()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            train_running += loss.item() * x.size(0)

        train_loss = train_running / len(train_loader.dataset)

        prior.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                indices = extract_indices(vqvae, x, y)
                logits = prior(indices, y)
                targets = indices.view(indices.shape[0], -1).long()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                val_running += loss.item() * x.size(0)

        val_loss = val_running / len(val_loader.dataset)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} "
            f"train CE={train_loss:.4f} val CE={val_loss:.4f} "
            f"lr={lr:.2e} es={es_counter}/{cfg.patience}"
        )

        if val_loss < best_val - cfg.es_min_delta:
            best_val = val_loss
            best_epoch = epoch
            es_counter = 0
            torch.save(
                {
                    "prior_state_dict": prior.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                cfg.ckpt_path,
            )
        else:
            es_counter += 1

        if es_counter >= cfg.patience:
            print(f"Early stopping prior training at epoch {epoch}.")
            break

    print(f"Best prior val CE: {best_val:.4f} at epoch {best_epoch}")
    return best_val, best_epoch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train class-conditional transformer prior over VQ-VAE code indices.")
    parser.add_argument("--data-dir", type=Path, default=Path("UrbanSound8K"))
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--vq-checkpoint", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--warmup-start-factor", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--spec-t", type=int, default=176)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--vq-num-embeddings", type=int, default=512)
    parser.add_argument("--vq-commitment-beta", type=float, default=0.25)
    parser.add_argument("--vq-ema-decay", type=float, default=0.99)
    parser.add_argument("--vq-ema-eps", type=float, default=1e-5)

    parser.add_argument("--prior-hidden-dim", type=int, default=256)
    parser.add_argument("--prior-layers", type=int, default=8)
    parser.add_argument("--prior-heads", type=int, default=8)
    parser.add_argument("--prior-dropout", type=float, default=0.1)
    parser.add_argument("--prior-max-seq-len", type=int, default=4096)

    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--run-prefix", type=str, default="transformer_prior")
    parser.add_argument("--run-idx", type=int, default=1)
    parser.add_argument("--prior-ckpt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _set_seed(args.seed)

    device = get_device()
    data_dir = resolve_urbansound_data_dir(args.data_dir)
    spec_dir = (args.spec_dir or (data_dir / "spectrograms")).resolve()
    metadata = load_metadata(data_dir)

    hp_for_name = {
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "spec_t": args.spec_t,
        "n_mels": args.n_mels,
        "n_fft": 0,
        "hop_length": 0,
        "vq_num_embeddings": args.vq_num_embeddings,
    }

    experiments_dir = (args.experiments_dir or (Path(__file__).resolve().parents[1] / "experiments" / "vq_prior")).resolve()
    run_name = make_run_name(run_prefix=args.run_prefix, hp=hp_for_name, run_idx=args.run_idx)
    auto_paths = make_run_paths(
        experiments_dir=experiments_dir,
        run_name=run_name,
        ckpt_filename="prior_best.pt",
        summary_filename="prior_summary.json",
    )

    prior_ckpt_path = args.prior_ckpt_path.resolve() if args.prior_ckpt_path is not None else auto_paths["ckpt_path"]
    summary_path = args.summary_path.resolve() if args.summary_path is not None else auto_paths["summary_path"]

    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Spec dir: {spec_dir}")
    print(f"VQ checkpoint: {args.vq_checkpoint}")
    print(f"Prior checkpoint output: {prior_ckpt_path}")
    print(f"Summary output: {summary_path}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    _, train_loader, val_loader = build_dataloaders(
        metadata=metadata,
        spec_dir=spec_dir,
        config=train_cfg,
        spec_t=args.spec_t,
    )

    vq_cfg = ModelConfig(
        n_classes=int(metadata["classID"].nunique()),
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        spec_h=args.n_mels,
        spec_t=args.spec_t,
        vq_num_embeddings=args.vq_num_embeddings,
        vq_commitment_beta=args.vq_commitment_beta,
        vq_ema_decay=args.vq_ema_decay,
        vq_ema_eps=args.vq_ema_eps,
    )

    vqvae = build_model(model_type="vqvae", config=vq_cfg)
    if not isinstance(vqvae, ConditionalVQVAE):
        raise RuntimeError("Failed to construct ConditionalVQVAE.")
    vqvae = vqvae.to(device)
    vqvae.load_state_dict(torch.load(args.vq_checkpoint, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    prior_cfg = PriorConfig(
        num_embeddings=args.vq_num_embeddings,
        n_classes=vq_cfg.n_classes,
        hidden_dim=args.prior_hidden_dim,
        n_layers=args.prior_layers,
        n_heads=args.prior_heads,
        dropout=args.prior_dropout,
        max_seq_len=args.prior_max_seq_len,
    )
    prior = build_prior(prior_cfg).to(device)

    cfg = PriorTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr=args.min_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        patience=args.patience,
        ckpt_path=prior_ckpt_path,
        summary_path=summary_path,
    )

    best_val, best_epoch = train_prior(
        prior=prior,
        vqvae=vqvae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )

    payload = {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "vq_checkpoint": str(args.vq_checkpoint),
        "prior_checkpoint": str(prior_ckpt_path),
        "prior_config": asdict(prior_cfg),
        "vq_model_config": asdict(vq_cfg),
        "train_config": asdict(cfg),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"Saved prior summary: {summary_path}")


if __name__ == "__main__":
    main()
