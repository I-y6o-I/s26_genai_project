from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import unpack_model_output


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.full_like(values[0], 1e-4)
    return values.std(dim=0).clamp_min(1e-4)


@torch.no_grad()
def fit_latent_stats(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> dict[str, torch.Tensor]:
    model.eval()

    mus: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    for x, labels in data_loader:
        x = x.to(device)
        labels = labels.to(device)
        _, _, mu, logvar = unpack_model_output(model(x, labels))
        if mu is None or logvar is None:
            raise RuntimeError("Latent statistics require VAE outputs with mu/logvar.")
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        labels_all.append(labels.cpu())

    mus_t = torch.cat(mus, dim=0)
    logvars_t = torch.cat(logvars, dim=0)
    labels_t = torch.cat(labels_all, dim=0)

    posterior_var = logvars_t.exp()
    aggregate_mean = mus_t.mean(dim=0)
    aggregate_std = torch.sqrt(mus_t.var(dim=0, unbiased=False) + posterior_var.mean(dim=0)).clamp_min(1e-4)

    class_mean = []
    class_std = []
    class_counts = []
    for class_id in range(n_classes):
        class_mask = labels_t == class_id
        class_counts.append(int(class_mask.sum().item()))
        class_mu = mus_t[class_mask]
        class_var = posterior_var[class_mask]
        if class_mu.shape[0] == 0:
            class_mean.append(aggregate_mean)
            class_std.append(aggregate_std)
            continue
        mean = class_mu.mean(dim=0)
        std = torch.sqrt(class_mu.var(dim=0, unbiased=False) + class_var.mean(dim=0)).clamp_min(1e-4)
        class_mean.append(mean)
        class_std.append(std)

    return {
        "aggregate_mean": aggregate_mean,
        "aggregate_std": aggregate_std,
        "class_mean": torch.stack(class_mean, dim=0),
        "class_std": torch.stack(class_std, dim=0),
        "mu_mean": mus_t.mean(dim=0),
        "mu_std": _safe_std(mus_t),
        "posterior_std_mean": torch.exp(0.5 * logvars_t).mean(dim=0),
        "class_counts": torch.tensor(class_counts, dtype=torch.long),
    }


def save_latent_stats(stats: dict[str, torch.Tensor], path: str | Path) -> Path:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out_path)
    return out_path


def load_latent_stats(path: str | Path, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    map_location = device if device is not None else "cpu"
    return torch.load(Path(path), map_location=map_location)


@torch.no_grad()
def generate_from_stats(
    model: torch.nn.Module,
    labels: torch.Tensor,
    latent_stats: dict[str, torch.Tensor],
    device: torch.device,
    mode: str = "class_posterior",
    temperature: float = 1.0,
) -> torch.Tensor:
    labels = labels.to(device)

    if mode == "posterior":
        mean = latent_stats["aggregate_mean"].to(device)
        std = latent_stats["aggregate_std"].to(device) * temperature
        z = mean.unsqueeze(0) + torch.randn(labels.shape[0], mean.shape[0], device=device) * std.unsqueeze(0)
        return model.decoder(z, labels)

    if mode == "class_posterior":
        mean = latent_stats["class_mean"].to(device)[labels]
        std = latent_stats["class_std"].to(device)[labels] * temperature
        z = mean + torch.randn_like(std) * std
        return model.decoder(z, labels)

    if mode == "mu_cluster":
        mean = latent_stats["class_mean"].to(device)[labels]
        std = latent_stats["mu_std"].to(device).unsqueeze(0) * temperature
        z = mean + torch.randn(labels.shape[0], mean.shape[1], device=device) * std
        return model.decoder(z, labels)

    raise ValueError(f"Unsupported latent sampling mode: {mode}")
