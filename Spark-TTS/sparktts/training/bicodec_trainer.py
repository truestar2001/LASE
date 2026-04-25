import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparktts.models.bicodec import BiCodec
from sparktts.training.adversarial import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MelSpecReconstructionLoss,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from sparktts.utils.audio import load_audio, stft
from sparktts.utils.file import load_config, read_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # 1) 일반 JSON array / dict 시도
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
    except json.JSONDecodeError:
        pass

    # 2) JSONL fallback
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL manifest at {path}, line {lineno}: {e}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object per line in {path}, line {lineno}, got {type(obj)}"
                )
            entries.append(obj)

    return entries


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        return _read_json_file(path)
    raise ValueError(f"Unsupported manifest extension: {path}")


def pad_1d_batch(items: Sequence[torch.Tensor]) -> torch.Tensor:
    max_len = max(x.shape[-1] for x in items)
    padded = []
    for x in items:
        if x.shape[-1] == max_len:
            padded.append(x)
            continue
        padded.append(F.pad(x, (0, max_len - x.shape[-1])))
    return torch.stack(padded, dim=0)


def crop_to_match_last_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = min(a.shape[-1], b.shape[-1])
    return a[..., :length], b[..., :length]


def build_reference_clip(wav: torch.Tensor, sample_rate: int, duration: float) -> torch.Tensor:
    ref_length = int(sample_rate * duration)
    if wav.shape[-1] >= ref_length:
        return wav[:ref_length]
    repeat = math.ceil(ref_length / max(1, wav.shape[-1]))
    return wav.repeat(repeat)[:ref_length]


class AudioManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        audio_key: str = "audio_filepath",
        base_dir: Optional[str] = None,
        volume_normalize: bool = False,
        segment_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        ref_segment_duration: float = 3.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.entries = load_manifest(self.manifest_path)
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.base_dir = Path(base_dir) if base_dir else None
        self.volume_normalize = volume_normalize
        self.segment_duration = segment_duration
        self.ref_segment_duration = ref_segment_duration
        self.entries = self._filter_by_duration(min_duration, max_duration)

    def _filter_by_duration(
        self,
        min_duration: Optional[float],
        max_duration: Optional[float],
    ) -> List[Dict[str, Any]]:
        if min_duration is None and max_duration is None:
            return self.entries

        filtered = []
        for entry in self.entries:
            duration = entry.get("duration")
            if duration is None:
                filtered.append(entry)
                continue
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            filtered.append(entry)
        return filtered

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_audio_path(self, entry: Dict[str, Any]) -> Path:
        wav_path = Path(entry[self.audio_key])
        if wav_path.is_absolute() or self.base_dir is None:
            return wav_path
        return self.base_dir / wav_path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        wav_path = self._resolve_audio_path(entry)
        wav = load_audio(
            wav_path,
            sampling_rate=self.sample_rate,
            volume_normalize=self.volume_normalize,
            segment_duration=self.segment_duration,
        )
        wav_tensor = torch.from_numpy(wav).float()
        ref_wav = build_reference_clip(
            wav_tensor,
            sample_rate=self.sample_rate,
            duration=self.ref_segment_duration,
        )
        return {
            "wav": wav_tensor,
            "ref_wav": ref_wav,
            "audio_filepath": str(wav_path),
        }


def collate_audio_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    wavs = pad_1d_batch([item["wav"] for item in batch])
    ref_wavs = pad_1d_batch([item["ref_wav"] for item in batch])
    return {
        "wav": wavs,
        "ref_wav": ref_wavs,
        "audio_filepath": [item["audio_filepath"] for item in batch],
    }


class FrozenWav2Vec2Frontend(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        hidden_state_indices: Sequence[int],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.model = Wav2Vec2Model.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_state_indices = tuple(hidden_state_indices)

    @torch.no_grad()
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            [wav.cpu().numpy() for wav in wavs],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        outputs = self.model(inputs.input_values.to(self.model.device), output_hidden_states=True)
        hidden_states = [outputs.hidden_states[idx] for idx in self.hidden_state_indices]
        return sum(hidden_states) / len(hidden_states)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, resolutions: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.resolutions = [tuple(map(int, resolution)) for resolution in resolutions]

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        total = pred_wav.new_tensor(0.0)
        for fft_size, hop_size, win_length in self.resolutions:
            window = torch.hann_window(win_length, device=pred_wav.device)
            pred_mag = stft(pred_wav, fft_size, hop_size, win_length, window)
            target_mag = stft(target_wav, fft_size, hop_size, win_length, window)
            total = total + F.l1_loss(pred_mag, target_mag)
            total = total + F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
        return total / len(self.resolutions)


def load_state_dict_for_training(path: str) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(path)
    if ckpt_path.suffix == ".safetensors":
        return load_safetensors(str(ckpt_path))

    pkg = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(pkg, dict) and "state_dict" in pkg:
        return pkg["state_dict"]
    if isinstance(pkg, dict):
        return pkg
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def build_bicodec_model(
    config_path: str,
    init_ckpt_path: Optional[str],
    strict_load: bool,
    quantizer_distance_loss_type: Optional[str] = None,
    use_codebook: Optional[bool] = None,
) -> Tuple[BiCodec, DictConfig]:
    model_config = load_config(config_path)
    model = BiCodec.from_config(model_config)
    if use_codebook is not None:
        model.use_codebook = bool(use_codebook)
    if quantizer_distance_loss_type:
        model.quantizer.distance_loss_type = quantizer_distance_loss_type
    if init_ckpt_path:
        state_dict = load_state_dict_for_training(init_ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
        if missing:
            print(f"[init] missing keys: {missing}")
        if unexpected:
            print(f"[init] unexpected keys: {unexpected}")
    return model, model_config


def resolve_training_paths(cfg: DictConfig) -> Dict[str, str]:
    pretrained_model_dir = cfg.model.get("pretrained_model_dir")
    model_config_path = cfg.model.get("config_path")
    init_ckpt_path = cfg.model.get("init_ckpt_path")
    feature_extractor_path = cfg.model.get("feature_extractor_path")

    if pretrained_model_dir is not None:
        pretrained_model_dir = Path(pretrained_model_dir)
        model_config_path = model_config_path or str(pretrained_model_dir / "BiCodec" / "config.yaml")
        init_ckpt_path = init_ckpt_path or str(pretrained_model_dir / "BiCodec" / "model.safetensors")
        feature_extractor_path = feature_extractor_path or str(pretrained_model_dir / "wav2vec2-large-xlsr-53")

    if model_config_path is None:
        raise ValueError("`model.config_path` or `model.pretrained_model_dir` is required")
    if feature_extractor_path is None:
        raise ValueError("`model.feature_extractor_path` or `model.pretrained_model_dir` is required")

    return {
        "model_config_path": model_config_path,
        "init_ckpt_path": init_ckpt_path,
        "feature_extractor_path": feature_extractor_path,
    }


def make_optimizer(cfg: DictConfig, params) -> AdamW:
    return AdamW(
        params,
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.optim.get("weight_decay", 0.0)),
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler_g: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: Optional[float],
    cfg: DictConfig,
    discriminator_state: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer_g.state_dict(),
        "scheduler": None if scheduler_g is None else scheduler_g.state_dict(),
        "scaler": scaler_g.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if discriminator_state is not None:
        payload["discriminator"] = discriminator_state
    torch.save(payload, checkpoint_path)


def maybe_load_resume(
    resume_path: Optional[str],
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler_g: GradScaler,
    mpd: Optional[nn.Module] = None,
    mrd: Optional[nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler_d: Optional[GradScaler] = None,
) -> Tuple[int, int, Optional[float]]:
    if not resume_path:
        return 0, 0, None

    pkg = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(pkg["state_dict"])
    optimizer_g.load_state_dict(pkg["optimizer"])
    if scheduler_g is not None and pkg.get("scheduler") is not None:
        scheduler_g.load_state_dict(pkg["scheduler"])
    if pkg.get("scaler") is not None:
        scaler_g.load_state_dict(pkg["scaler"])

    disc_state = pkg.get("discriminator")
    if disc_state is not None:
        if mpd is not None and disc_state.get("mpd") is not None:
            mpd.load_state_dict(disc_state["mpd"])
        if mrd is not None and disc_state.get("mrd") is not None:
            mrd.load_state_dict(disc_state["mrd"])
        if optimizer_d is not None and disc_state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(disc_state["optimizer_d"])
        if scheduler_d is not None and disc_state.get("scheduler_d") is not None:
            scheduler_d.load_state_dict(disc_state["scheduler_d"])
        if scaler_d is not None and disc_state.get("scaler_d") is not None:
            scaler_d.load_state_dict(disc_state["scaler_d"])
    return int(pkg.get("epoch", 0)), int(pkg.get("global_step", 0)), pkg.get("best_val_loss")


def compute_reconstruction_losses(
    outputs: Dict[str, torch.Tensor],
    feat_target: torch.Tensor,
    mel_loss_fn: MelSpecReconstructionLoss,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: DictConfig,
) -> Dict[str, torch.Tensor]:
    pred_wav, target_wav = crop_to_match_last_dim(outputs["recons"], outputs["audios"])
    pred_feat, target_feat = crop_to_match_last_dim(outputs["pred_feat"], feat_target.transpose(1, 2))

    waveform_l1 = F.l1_loss(pred_wav, target_wav)
    feature_l1 = F.l1_loss(pred_feat, target_feat)
    mel = mel_loss_fn(pred_wav.squeeze(1), target_wav.squeeze(1))
    spectral = stft_loss_fn(pred_wav.squeeze(1), target_wav.squeeze(1))
    vq_loss = outputs["vq_loss"]

    total = (
        float(cfg.loss.waveform_l1_weight) * waveform_l1
        + float(cfg.loss.feature_l1_weight) * feature_l1
        + float(cfg.loss.mel_weight) * mel
        + float(cfg.loss.stft_weight) * spectral
        + float(cfg.loss.vq_weight) * vq_loss
    )
    return {
        "total": total,
        "waveform_l1": waveform_l1.detach(),
        "feature_l1": feature_l1.detach(),
        "mel": mel.detach(),
        "stft": spectral.detach(),
        "vq": vq_loss.detach(),
        "perplexity": outputs["perplexity"].detach(),
        "active_num": outputs["cluster_size"].detach(),
    }


def average_discriminator_loss(
    loss_fn: DiscriminatorLoss,
    real_outputs: List[torch.Tensor],
    fake_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss, real_losses, _ = loss_fn(real_outputs, fake_outputs)
    return loss / max(len(real_losses), 1)


def average_generator_loss(
    loss_fn: GeneratorLoss,
    disc_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss, sub_losses = loss_fn(disc_outputs)
    return loss / max(len(sub_losses), 1)


def average_feature_matching_loss(
    loss_fn: FeatureMatchingLoss,
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]],
) -> torch.Tensor:
    return loss_fn(fmap_r, fmap_g) / max(len(fmap_r), 1)


def evaluate(
    model: nn.Module,
    frontend: FrozenWav2Vec2Frontend,
    dataloader: DataLoader,
    mel_loss_fn: MelSpecReconstructionLoss,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: DictConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            wav = batch["wav"].to(device)
            ref_wav = batch["ref_wav"].to(device)
            feat = frontend(wav)
            outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
            losses = compute_reconstruction_losses(outputs, feat, mel_loss_fn, stft_loss_fn, cfg)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1

    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_bicodec(cfg: DictConfig) -> None:
    set_seed(int(cfg.get("seed", 42)))
    paths = resolve_training_paths(cfg)
    output_dir = Path(cfg.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "train_config.yaml")

    device_name = cfg.trainer.get("device")
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("current_device:", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    model, model_config = build_bicodec_model(
        config_path=paths["model_config_path"],
        init_ckpt_path=paths["init_ckpt_path"],
        strict_load=bool(cfg.model.get("strict_load", False)),
        quantizer_distance_loss_type=cfg.loss.get("vq_distance_loss_type"),
        use_codebook=cfg.model.get("use_codebook"),
    )
    model = model.to(device)

    frontend = FrozenWav2Vec2Frontend(
        model_name_or_path=paths["feature_extractor_path"],
        hidden_state_indices=cfg.feature_extractor.hidden_state_indices,
        device=device,
    )

    sample_rate = int(model_config.get("sample_rate", model_config.get("audio_tokenizer", {}).get("sample_rate", 16000)))
    if sample_rate != 16000:
        raise ValueError(f"Training frontend currently expects 16000 Hz audio, got {sample_rate}")

    train_dataset = AudioManifestDataset(
        manifest_path=cfg.data.train_manifest,
        sample_rate=sample_rate,
        audio_key=cfg.data.get("audio_key", "audio_filepath"),
        base_dir=cfg.data.get("base_dir"),
        volume_normalize=bool(cfg.data.get("volume_normalize", False)),
        segment_duration=cfg.data.get("segment_duration"),
        min_duration=cfg.data.get("min_duration"),
        max_duration=cfg.data.get("max_duration"),
        ref_segment_duration=float(model_config.get("ref_segment_duration", cfg.data.get("ref_segment_duration", 3.0))),
    )
    valid_manifest = cfg.data.get("valid_manifest")
    valid_dataset = None
    if valid_manifest:
        valid_dataset = AudioManifestDataset(
            manifest_path=valid_manifest,
            sample_rate=sample_rate,
            audio_key=cfg.data.get("audio_key", "audio_filepath"),
            base_dir=cfg.data.get("base_dir"),
            volume_normalize=bool(cfg.data.get("volume_normalize", False)),
            segment_duration=cfg.data.get("segment_duration"),
            min_duration=cfg.data.get("min_duration"),
            max_duration=cfg.data.get("max_duration"),
            ref_segment_duration=float(model_config.get("ref_segment_duration", cfg.data.get("ref_segment_duration", 3.0))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=True,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio_batch,
        drop_last=False,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(cfg.trainer.eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.trainer.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=collate_audio_batch,
            drop_last=False,
        )

    sample_rate_for_losses = int(
        model_config.get("sample_rate", model_config.get("audio_tokenizer", {}).get("sample_rate", 16000))
    )
    mel_loss_fn = MelSpecReconstructionLoss(
        sample_rate=sample_rate_for_losses,
        n_fft=int(cfg.loss.mel_n_fft),
        hop_length=int(cfg.loss.mel_hop_length),
        n_mels=int(cfg.loss.mel_n_mels),
        f_min=int(cfg.loss.get("mel_f_min", 0)),
        f_max=int(cfg.loss.get("mel_f_max", sample_rate_for_losses // 2)),
    ).to(device)
    stft_loss_fn = MultiResolutionSTFTLoss(cfg.loss.stft_resolutions)

    mpd = MultiPeriodDiscriminator(periods=tuple(cfg.discriminator.get("periods", [2, 3, 5, 7, 11]))).to(device)
    mrd = MultiResolutionDiscriminator(fft_sizes=tuple(cfg.discriminator.get("fft_sizes", [2048, 1024, 512]))).to(device)
    gen_loss_fn = GeneratorLoss()
    disc_loss_fn = DiscriminatorLoss()
    feat_matching_loss_fn = FeatureMatchingLoss()

    optimizer = make_optimizer(cfg, model.parameters())
    disc_optim_cfg = cfg.get("disc_optim", cfg.optim)
    optimizer_d = AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=float(disc_optim_cfg.lr),
        betas=tuple(disc_optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(disc_optim_cfg.get("weight_decay", 0.0)),
    )
    max_steps = int(cfg.trainer.max_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=float(cfg.optim.get("min_lr", 0.0)))
    scheduler_d = CosineAnnealingLR(
        optimizer_d,
        T_max=max_steps,
        eta_min=float(disc_optim_cfg.get("min_lr", 0.0)),
    )
    use_amp = bool(cfg.trainer.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)

    start_epoch, global_step, best_val_loss = maybe_load_resume(
        cfg.trainer.get("resume_from"),
        model,
        optimizer,
        scheduler,
        scaler,
        mpd=mpd,
        mrd=mrd,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        scaler_d=scaler_d,
    )

    grad_accum_steps = int(cfg.trainer.get("grad_accum_steps", 1))
    log_every = int(cfg.trainer.get("log_every_steps", 10))
    save_every = int(cfg.trainer.get("save_every_steps", 1000))
    eval_every = int(cfg.trainer.get("eval_every_steps", save_every))
    max_epochs = int(cfg.trainer.get("max_epochs", 1))
    pretrain_mel_steps = int(cfg.trainer.get("pretrain_mel_steps", 0))

    print(f"[train] device={device} train_items={len(train_dataset)} valid_items={0 if valid_dataset is None else len(valid_dataset)}")
    print(f"[train] output_dir={output_dir}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, max_epochs):
        for batch in train_loader:
            wav = batch["wav"].to(device)
            ref_wav = batch["ref_wav"].to(device)
            feat = frontend(wav)
            train_discriminator = global_step >= pretrain_mel_steps

            if train_discriminator:
                with torch.no_grad():
                    disc_outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
                    fake_wav = crop_to_match_last_dim(disc_outputs["recons"], disc_outputs["audios"])[0].squeeze(1)
                    real_wav = crop_to_match_last_dim(disc_outputs["recons"], disc_outputs["audios"])[1].squeeze(1)

                with autocast(device_type=device.type, enabled=use_amp):
                    real_score_mp, fake_score_mp, _, _ = mpd(real_wav, fake_wav)
                    real_score_mrd, fake_score_mrd, _, _ = mrd(real_wav, fake_wav)
                    loss_disc_mp = average_discriminator_loss(disc_loss_fn, real_score_mp, fake_score_mp)
                    loss_disc_mrd = average_discriminator_loss(disc_loss_fn, real_score_mrd, fake_score_mrd)
                    disc_loss = (
                        float(cfg.loss.discriminator_mp_weight) * loss_disc_mp
                        + float(cfg.loss.discriminator_mrd_weight) * loss_disc_mrd
                    )

                scaler_d.scale(disc_loss).backward()
                if (global_step + 1) % grad_accum_steps == 0:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        list(mpd.parameters()) + list(mrd.parameters()),
                        float(cfg.trainer.get("clip_grad_norm", 1.0)),
                    )
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
                    scheduler_d.step()

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
                recon_losses = compute_reconstruction_losses(outputs, feat, mel_loss_fn, stft_loss_fn, cfg)
                pred_wav, target_wav = crop_to_match_last_dim(outputs["recons"], outputs["audios"])
                fake_wav = pred_wav.squeeze(1)
                real_wav = target_wav.squeeze(1)

                loss_gen_mp = fake_wav.new_tensor(0.0)
                loss_gen_mrd = fake_wav.new_tensor(0.0)
                loss_fm_mp = fake_wav.new_tensor(0.0)
                loss_fm_mrd = fake_wav.new_tensor(0.0)

                if train_discriminator:
                    _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = mpd(real_wav, fake_wav)
                    _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(real_wav, fake_wav)
                    loss_gen_mp = average_generator_loss(gen_loss_fn, gen_score_mp)
                    loss_gen_mrd = average_generator_loss(gen_loss_fn, gen_score_mrd)
                    loss_fm_mp = average_feature_matching_loss(feat_matching_loss_fn, fmap_rs_mp, fmap_gs_mp)
                    loss_fm_mrd = average_feature_matching_loss(feat_matching_loss_fn, fmap_rs_mrd, fmap_gs_mrd)

                total_gen_loss = (
                    recon_losses["total"]
                    + float(cfg.loss.generator_mp_weight) * loss_gen_mp
                    + float(cfg.loss.generator_mrd_weight) * loss_gen_mrd
                    + float(cfg.loss.feature_matching_mp_weight) * loss_fm_mp
                    + float(cfg.loss.feature_matching_mrd_weight) * loss_fm_mrd
                )
                loss = total_gen_loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.trainer.get("clip_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if global_step % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    "[train] "
                    f"epoch={epoch} step={global_step} "
                    f"loss={float(total_gen_loss):.4f} "
                    f"mel={float(recon_losses['mel']):.4f} "
                    f"wav_l1={float(recon_losses['waveform_l1']):.4f} "
                    f"feat_l1={float(recon_losses['feature_l1']):.4f} "
                    f"stft={float(recon_losses['stft']):.4f} "
                    f"vq={float(recon_losses['vq']):.4f} "
                    f"gan_mp={float(loss_gen_mp):.4f} "
                    f"gan_mrd={float(loss_gen_mrd):.4f} "
                    f"fm_mp={float(loss_fm_mp):.4f} "
                    f"fm_mrd={float(loss_fm_mrd):.4f} "
                    f"disc={float(disc_loss) if train_discriminator else 0.0:.4f} "
                    f"ppl={float(recon_losses['perplexity']):.2f} "
                    f"active={float(recon_losses['active_num']):.2f} "
                    f"lr={current_lr:.7f}"
                )

            if global_step % save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    cfg,
                    discriminator_state={
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "scaler_d": scaler_d.state_dict(),
                    },
                )

            if valid_loader is not None and global_step % eval_every == 0:
                metrics = evaluate(model, frontend, valid_loader, mel_loss_fn, stft_loss_fn, cfg, device)
                print(
                    "[valid] "
                    f"step={global_step} "
                    f"loss={metrics['total']:.4f} "
                    f"mel={metrics['mel']:.4f} "
                    f"wav_l1={metrics['waveform_l1']:.4f} "
                    f"feat_l1={metrics['feature_l1']:.4f} "
                    f"stft={metrics['stft']:.4f} "
                    f"vq={metrics['vq']:.4f}"
                )
                val_loss = metrics["total"]
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "checkpoints" / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_val_loss,
                        cfg,
                        discriminator_state={
                            "mpd": mpd.state_dict(),
                            "mrd": mrd.state_dict(),
                            "optimizer_d": optimizer_d.state_dict(),
                            "scheduler_d": scheduler_d.state_dict(),
                            "scaler_d": scaler_d.state_dict(),
                        },
                    )

            if global_step >= max_steps:
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    cfg,
                    discriminator_state={
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "scaler_d": scaler_d.state_dict(),
                    },
                )
                print(f"[done] reached max_steps={max_steps}")
                return

        save_checkpoint(
            output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            global_step,
            best_val_loss,
            cfg,
            discriminator_state={
                "mpd": mpd.state_dict(),
                "mrd": mrd.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "scaler_d": scaler_d.state_dict(),
            },
        )

    save_checkpoint(
        output_dir / "checkpoints" / "last.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        max_epochs,
        global_step,
        best_val_loss,
        cfg,
        discriminator_state={
            "mpd": mpd.state_dict(),
            "mrd": mrd.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "scaler_d": scaler_d.state_dict(),
        },
    )
