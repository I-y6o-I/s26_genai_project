from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    import librosa
except Exception:  # pragma: no cover - optional at import time
    librosa = None


DEFAULT_SR = 22050
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def _require_librosa() -> None:
    if librosa is None:
        raise ModuleNotFoundError(
            "librosa is required for audio preprocessing/evaluation. Install with `pip install librosa`."
        )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_urbansound_data_dir(data_dir: str | Path) -> Path:
    requested = Path(data_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        requested,
        requested / "UrbanSound8K",
        requested.parent / "UrbanSound8K",
        project_root / "UrbanSound8K",
        project_root.parent / "UrbanSound8K",
        Path.cwd().resolve() / "UrbanSound8K",
    ]
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if (candidate / "metadata" / "UrbanSound8K.csv").is_file():
            return candidate

    checked = "\n".join(str(path / "metadata" / "UrbanSound8K.csv") for path in unique_candidates)
    raise FileNotFoundError(
        "UrbanSound8K.csv was not found. Checked:\n"
        f"{checked}\n"
        "Pass --data-dir pointing to the folder that contains `metadata/UrbanSound8K.csv`."
    )


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    data_dir = resolve_urbansound_data_dir(data_dir)
    csv_path = data_dir / "metadata" / "UrbanSound8K.csv"
    metadata = pd.read_csv(csv_path)
    metadata["length"] = metadata["end"] - metadata["start"]
    return metadata


def unpack_audio(src_root: str | Path, dest_dir: str | Path, move_files: bool = False) -> None:
    """Flatten nested UrbanSound folders into a single directory."""
    src_root = Path(src_root).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for root, _, files in os.walk(src_root):
        root_path = Path(root).resolve()
        for filename in files:
            src_file = root_path / filename
            if not src_file.is_file() or src_file.name.startswith("."):
                continue
            target = dest_dir / src_file.name
            if move_files:
                shutil.move(str(src_file), str(target))
            else:
                shutil.copy2(src_file, target)
            copied += 1
    action = "moved" if move_files else "copied"
    print(f"{action.capitalize()} {copied} audio files -> {dest_dir}")


def save_log_mel_spectrogram(
    metadata: pd.DataFrame,
    audio_dir: str | Path,
    output_dir: str | Path,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> None:
    _require_librosa()
    audio_dir = Path(audio_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(metadata["slice_file_name"], desc="Saving log-mel"):
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            print(f"[SKIP] not found: {audio_path}")
            continue

        wave, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=wave,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        np.save(output_dir / f"{Path(filename).stem}.npy", log_mel)

    print(f"Saved spectrograms to {output_dir}")


def denorm_log_mel(spec_t: torch.Tensor) -> np.ndarray:
    """(B,1,H,T) in [-1,1] -> (B,H,T) in dB scale around [-80, 0]."""
    return (spec_t.squeeze(1).cpu().numpy() * 40.0) - 40.0


def spec_to_waveform(
    log_mel_db: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_iter: int = 60,
) -> np.ndarray:
    _require_librosa()
    mel_power = librosa.db_to_power(log_mel_db)
    mel_to_stft = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=log_mel_db.shape[0])
    stft_mag = np.dot(np.linalg.pinv(mel_to_stft), mel_power).clip(0)
    return librosa.griffinlim(stft_mag, n_iter=n_iter, hop_length=hop_length)
