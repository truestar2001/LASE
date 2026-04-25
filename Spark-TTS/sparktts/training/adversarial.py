from typing import List, Tuple

import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))


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
        return torch.nn.functional.l1_loss(mel, mel_hat)


class GeneratorLoss(nn.Module):
    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            current = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(current)
            loss = loss + current
        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        real_losses = []
        fake_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            real_loss = torch.mean(torch.clamp(1 - dr, min=0))
            fake_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss = loss + real_loss + fake_loss
            real_losses.append(real_loss)
            fake_losses.append(fake_loss)
        return loss, real_losses, fake_losses


class FeatureMatchingLoss(nn.Module):
    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss = loss + torch.mean(torch.abs(rl - gl))
        return loss


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(period=p) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
            ]
        )
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.unsqueeze(1)
        fmap = []
        batch_size, channels, total_length = x.shape
        if total_length % self.period != 0:
            n_pad = self.period - (total_length % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            total_length = total_length + n_pad
        x = x.view(batch_size, channels, total_length // self.period, self.period)

        for idx, layer in enumerate(self.convs):
            x = torch.nn.functional.leaky_relu(layer(x), self.lrelu_slope)
            if idx > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, fft_sizes: Tuple[int, ...] = (2048, 1024, 512)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorR(window_length=w) for w in fft_sizes])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    ):
        super().__init__()
        self.spec_fn = Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            win_length=window_length,
            power=None,
        )
        n_fft = window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.band_convs = nn.ModuleList([self._make_stack(channels) for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    @staticmethod
    def _make_stack(channels: int) -> nn.ModuleList:
        return nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x - x.mean(dim=-1, keepdim=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b f t c -> b c t f")
        return [x[..., start:end] for start, end in self.bands]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bands = self.spectrogram(x)
        fmap = []
        band_outputs = []
        for band, stack in zip(x_bands, self.band_convs):
            for idx, layer in enumerate(stack):
                band = torch.nn.functional.leaky_relu(layer(band), 0.1)
                if idx > 0:
                    fmap.append(band)
            band_outputs.append(band)
        x = torch.cat(band_outputs, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        return x, fmap
