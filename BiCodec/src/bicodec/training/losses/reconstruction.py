from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig
from torch import nn

from bicodec.utils.audio import stft


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))


def crop_to_match_last_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = min(a.shape[-1], b.shape[-1])
    return a[..., :length], b[..., :length]


class MelSpecReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: int = 0,
        f_max: int = 8000,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=1,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))
        return F.l1_loss(mel, mel_hat)


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


def compute_reconstruction_losses(
    outputs: Dict[str, torch.Tensor],
    feat_target: torch.Tensor,
    mel_loss_fn: nn.Module,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: DictConfig,
) -> Dict[str, torch.Tensor]:
    pred_wav, target_wav = crop_to_match_last_dim(outputs["recons"], outputs["audios"])
    pred_feat, target_feat = crop_to_match_last_dim(
        outputs["pred_feat"],
        feat_target.transpose(1, 2),
    )

    waveform_l1 = F.l1_loss(pred_wav, target_wav)
    feature_l1 = F.l1_loss(pred_feat, target_feat)
    mel_loss = mel_loss_fn(pred_wav.squeeze(1), target_wav.squeeze(1))
    stft_loss = stft_loss_fn(pred_wav.squeeze(1), target_wav.squeeze(1))
    vq_loss = outputs["vq_loss"]
    vq_weight = float(cfg.loss.vq_weight)
    skip_vq_loss = vq_weight == 0.0
    vq_term = pred_wav.new_tensor(0.0) if skip_vq_loss else vq_weight * vq_loss

    total = (
        float(cfg.loss.waveform_l1_weight) * waveform_l1
        + float(cfg.loss.feature_l1_weight) * feature_l1
        + float(cfg.loss.mel_weight) * mel_loss
        + float(cfg.loss.stft_weight) * stft_loss
        + vq_term
    )
    return {
        "total": total,
        "waveform_l1": waveform_l1.detach(),
        "feature_l1": feature_l1.detach(),
        "mel": mel_loss.detach(),
        "stft": stft_loss.detach(),
        "vq": pred_wav.new_tensor(0.0).detach() if skip_vq_loss else vq_loss.detach(),
        "perplexity": outputs["perplexity"].detach(),
        "active_num": outputs["cluster_size"].detach(),
    }
