# UrbanSound8K Generative Audio Models

This repository contains the code and reports for a course project on class-conditional audio generation from UrbanSound8K log-mel spectrograms. The main entry points are:

- `src/train.py` for preprocessing and training `ae`, `vae`, or `vqvae`
- `src/eval.py` for reconstruction/generation metrics and qualitative exports
- `src/train_prior.py` for training a transformer prior over VQ-VAE code indices

## 1. Environment setup

Recommended: Python 3.10+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your platform needs a custom PyTorch wheel, install `torch` and `torchvision` first from the official PyTorch instructions, then run:

```bash
pip install -r requirements.txt
```

## 2. Dataset layout

The code expects the UrbanSound8K directory to contain:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── metadata/
    └── UrbanSound8K.csv
```

During preprocessing, the scripts can create:

- `UrbanSound8K/audio_unpacked/` with flattened audio files
- `UrbanSound8K/spectrograms/` with `.npy` log-mel spectrograms

By default, all scripts look for a folder named `UrbanSound8K/` in the project root. You can override that with `--data-dir`.

## 3. Preprocessing

Run preprocessing once before training:

```bash
python src/train.py \
  --data-dir UrbanSound8K \
  --prepare-audio \
  --prepare-specs \
  --num-workers 8
```

This will:

- flatten `audio/fold*/` into `audio_unpacked/`
- generate log-mel spectrogram `.npy` files into `spectrograms/`

If you already prepared one of those steps, you can run only the missing one by keeping just `--prepare-audio` or just `--prepare-specs`.

## 4. Training

### Conditional AE

```bash
python src/train.py \
  --data-dir UrbanSound8K \
  --model-type ae \
  --epochs 150 \
  --batch-size 64 \
  --lr 3e-4 \
  --num-workers 8
```

### Conditional VAE

```bash
python src/train.py \
  --data-dir UrbanSound8K \
  --model-type vae \
  --epochs 150 \
  --batch-size 64 \
  --lr 3e-4 \
  --beta-kl 0.001 \
  --num-workers 8
```

### Conditional VQ-VAE

```bash
python src/train.py \
  --data-dir UrbanSound8K \
  --model-type vqvae \
  --epochs 150 \
  --batch-size 64 \
  --lr 3e-4 \
  --vq-num-embeddings 512 \
  --vq-commitment-beta 0.25 \
  --num-workers 8
```

Useful optional flags:

- `--spec-dir PATH` to use a custom spectrogram directory
- `--experiments-dir PATH` to change where outputs are saved
- `--run-prefix NAME --run-idx N` to control run names
- `--sanity-overfit --overfit-samples 32` for a quick debugging run

Training outputs are saved under `experiments/<model_type>/<run_name>/` and include:

- `checkpoints/*_best.pt`
- `tb/` TensorBoard logs
- `train_summary.json`
- `latent_stats.pt` for VAE runs

To inspect TensorBoard logs:

```bash
tensorboard --logdir experiments
```

## 5. Evaluation

Replace the checkpoint path and hyperparameters below with the values used for the run you want to evaluate.

### Evaluate AE or VAE

```bash
python src/eval.py \
  --data-dir UrbanSound8K \
  --model-type vae \
  --checkpoint experiments/vae/<run_name>/checkpoints/vae_best.pt \
  --latent-dim 64 \
  --embed-dim 32 \
  --base-ch 32 \
  --n-mels 128 \
  --spec-t 176 \
  --batch-size 64 \
  --qualitative-count 4
```

For VAE generation modes other than the default prior sampling, use one of:

- `--sampling-mode posterior`
- `--sampling-mode class_posterior`
- `--sampling-mode mu_cluster`

If needed, you can pass latent statistics explicitly with `--latent-stats PATH`.

### Evaluate VQ-VAE

For meaningful VQ-VAE generation, provide a trained prior checkpoint:

```bash
python src/eval.py \
  --data-dir UrbanSound8K \
  --model-type vqvae \
  --checkpoint experiments/vqvae/<run_name>/checkpoints/vqvae_best.pt \
  --vq-prior-checkpoint experiments/vq_prior/<prior_run>/checkpoints/prior_best.pt \
  --latent-dim 128 \
  --embed-dim 32 \
  --base-ch 32 \
  --vq-num-embeddings 512 \
  --vq-commitment-beta 0.25 \
  --n-mels 128 \
  --spec-t 176 \
  --batch-size 64
```

Evaluation writes results to an `inference/` directory inside the inferred run folder unless `--output-dir` is provided. Main outputs:

- `metrics.json`
- `qualitative/` spectrogram images
- `qualitative/` generated, reconstructed, and ground-truth `.wav` files

## 6. Training the VQ-VAE prior

Train the prior after you already have a VQ-VAE checkpoint:

```bash
python src/train_prior.py \
  --data-dir UrbanSound8K \
  --vq-checkpoint experiments/vqvae/<run_name>/checkpoints/vqvae_best.pt \
  --latent-dim 128 \
  --embed-dim 32 \
  --base-ch 32 \
  --vq-num-embeddings 512 \
  --vq-commitment-beta 0.25 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8
```

Prior outputs are saved under `experiments/vq_prior/<run_name>/` by default and include:

- `checkpoints/prior_best.pt`
- `prior_summary.json`

## 7. Reports and notebook

- Reports are in `reports/`
- Hyperparameter exploration notebook: `notebooks/hparam_experiments.ipynb`

## 8. Reproducibility notes

- The scripts use a fixed default seed of `42`
- GPU is used automatically when CUDA is available
- For evaluation, reconstruction quality depends on Griffin-Lim inversion settings such as `--griffin-iters`
