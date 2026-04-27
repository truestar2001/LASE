# BiCodec

Standalone BiCodec training and inference repository extracted from the Spark-TTS codebase, with Spark-TTS-specific runtime pieces removed and the package reorganized into a standard `src/` layout.

## Scope

- BiCodec model definition
- BiCodec training loop and adversarial losses
- BiCodec tokenizer for local inference
- Experiment configs for `exp1`, `exp2`, and `exp3`

## Repository Layout

```text
BiCodec/
├── configs/
│   ├── base/
│   │   └── train_bicodec.yaml
│   ├── train_bicodec_exp1.yaml
│   ├── train_bicodec_exp2.yaml
│   └── train_bicodec_exp3.yaml
├── src/bicodec/
│   ├── cli/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── modules/
├── tests/
├── train_bicodec.sh
├── train_bicodec_ddp.sh
├── pyproject.toml
└── LICENSE
```

## Package Boundaries

- `bicodec.data`: manifest parsing, dataset definition, collate functions
- `bicodec.models`: `BiCodec` and inference tokenizer entrypoints
- `bicodec.training`: training engine, checkpointing, distributed helpers, freeze policy, losses
- `bicodec.modules`: neural network building blocks kept close to the original implementation
- `bicodec.utils`: config loading, artifact layout resolution, lightweight JSONL IO

## Artifact Layout

The repo expects a pretrained artifact root with this structure:

```text
artifacts/pretrained/bicodec-base/
├── bicodec/
│   ├── config.yaml
│   └── model.safetensors
└── wav2vec2-large-xlsr-53/
```

Legacy `BiCodec/` directories are still accepted for compatibility.

## Install

```bash
cd BiCodec
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

The default configs point to `data/manifests/voxceleb2_train.jsonl` and
`data/manifests/voxceleb2_valid.jsonl`.

To regenerate a speaker-stratified split from a source manifest:

```bash
bicodec-split-manifest \
  --input-manifest ../Spark-TTS/manifests/voxceleb2.jsonl \
  --train-output data/manifests/voxceleb2_train.jsonl \
  --valid-output data/manifests/voxceleb2_valid.jsonl \
  --valid-per-speaker 3
```

Run from the repo root:

```bash
./train_bicodec.sh
```

DDP:

```bash
./train_bicodec_ddp.sh
```

Edit the variables at the top of each script before running.
- `train_bicodec.sh`: set `CONFIG` and `GPU_IDS`
- `train_bicodec_ddp.sh`: set `CONFIG`, `GPU_IDS`, `NPROC_PER_NODE`, and `MASTER_PORT`
- Example: `GPU_IDS="0"` for single GPU, `GPU_IDS="0,1"` for two GPUs

### `exp1`

- Frozen modules: `encoder`, `quantizer`, `speaker_encoder`, `mel_transformer`
- Trainable modules: `prenet`, `postnet`, `decoder`
- Semantic path: `wav -> wav2vec2 frontend -> encoder -> quantizer -> z_q`
- Speaker path: `ref_wav -> mel -> speaker_encoder -> x_vector`
- Reconstruction path: `z_q` and detached `x_vector` are used as decoder conditions to reconstruct `wav_recon`
- Summary: keep semantic and speaker representations fixed, and train only the decoder stack for waveform reconstruction

### `exp2`

- Frozen modules: `encoder`, `quantizer`, `speaker_encoder`, `mel_transformer`
- Trainable modules: `speaker_adapter`, `speaker_classifier`, `prenet`, `postnet`, `decoder`
- Semantic path: `wav -> wav2vec2 frontend -> encoder -> quantizer -> z_q`
- Speaker path: `ref_wav -> mel -> speaker_encoder -> x_vector -> speaker_adapter`
- Reconstruction/classification path: the adapted speaker vector is shared by both the decoder condition and the speaker classifier
- Loss: reconstruction + GAN loss + speaker classification loss
- Summary: instead of using frozen `x_vector` directly, pass it through a small adapter so reconstruction and speaker classification learn from the same shared representation

## Inference

Tokenize an audio file:

```bash
bicodec-tokenize \
  --artifacts-dir artifacts/pretrained/bicodec-base \
  --audio-path sample.wav \
  --output outputs/sample_tokens.pt
```

Reconstruct from saved tokens:

```bash
bicodec-reconstruct \
  --artifacts-dir artifacts/pretrained/bicodec-base \
  --tokens-path outputs/sample_tokens.pt \
  --output-wav outputs/sample_recon.wav
```

## Experiment Configs

- `exp1`: freeze semantic/VQ/speaker encoder and train decoder stack
- `exp2`: `exp1` plus shared speaker adapter and speaker classification loss
- `exp3`: same-wave speaker branch, ECAPA latent resized to semantic `T`, then adapter plus original pool/project path

## Notes

- The repo keeps the extracted module implementations close to the original source to reduce behavior drift.
- The main cleanup is structural: package boundaries, dataset/trainer modularization, config inheritance, artifact layout resolution, and standalone CLI entrypoints.
