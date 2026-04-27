# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from bicodec.utils.config import load_config
from bicodec.modules.speaker.speaker_encoder import SpeakerEncoder
from bicodec.modules.encoder_decoder.feat_encoder import Encoder
from bicodec.modules.encoder_decoder.feat_decoder import Decoder
from bicodec.modules.encoder_decoder.wave_generator import WaveGenerator
from bicodec.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize


class SpeakerAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out_proj(self.dropout(self.act(self.in_proj(x))))


class SpeakerTimeAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("SpeakerTimeAdapter requires num_layers >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("SpeakerTimeAdapter requires an odd kernel_size >= 1")

        hidden_dim = int(hidden_dim or input_dim)
        padding = kernel_size // 2

        blocks = []
        in_dim = input_dim
        for _ in range(num_layers):
            block = [
                nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*block))
            in_dim = hidden_dim

        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.res_scale = nn.Parameter(torch.full((1,), 1e-3))

    def forward(self, ecapa_latent: torch.Tensor, target_length: int) -> torch.Tensor:
        if ecapa_latent.shape[-1] != target_length:
            ecapa_latent = F.interpolate(
                ecapa_latent,
                size=target_length,
                mode="linear",
                align_corners=False,
            )
        adapted = self.blocks(ecapa_latent)
        adapted = self.out_proj(adapted)
        return ecapa_latent + self.res_scale * adapted


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        speaker_adapter: Optional[nn.Module] = None,
        speaker_time_condition: Optional[nn.Module] = None,
        speaker_input_key: str = "ref_wav",
        detach_speaker_time_input: bool = False,
        use_codebook: bool = True,
        **kwargs
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            speaker_encoder (nn.Module): Speaker encoder module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.speaker_adapter = speaker_adapter
        self.prenet = prenet
        self.postnet = postnet
        self.speaker_time_condition = speaker_time_condition
        self.speaker_input_key = str(speaker_input_key)
        self.detach_speaker_time_input = bool(detach_speaker_time_input)
        self.use_codebook = use_codebook
        self.init_mel_transformer(mel_params)

    @staticmethod
    def _resolve_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(config, DictConfig):
            config = dict(config)
        if "audio_tokenizer" in config:
            config = config["audio_tokenizer"]
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BiCodec":
        config = cls._resolve_model_config(config)
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])
        speaker_input_key = str(config.get("speaker_input_key", "ref_wav"))
        speaker_adapter = None
        speaker_adapter_cfg = config.get("speaker_adapter")
        if speaker_adapter_cfg is None:
            speaker_adapter_cfg = config.get("speaker_conditioner")
        if isinstance(speaker_adapter_cfg, DictConfig):
            speaker_adapter_cfg = OmegaConf.to_container(
                speaker_adapter_cfg,
                resolve=True,
            )
        if isinstance(speaker_adapter_cfg, dict) and bool(
            speaker_adapter_cfg.get("enabled", False)
        ):
            speaker_adapter = SpeakerAdapter(
                input_dim=int(config["speaker_encoder"].get("out_dim", 512)),
                hidden_dim=speaker_adapter_cfg.get("hidden_dim"),
                dropout=float(speaker_adapter_cfg.get("dropout", 0.0)),
            )

        speaker_time_condition_cfg = config.get("speaker_time_adapter")
        if speaker_time_condition_cfg is None:
            speaker_time_condition_cfg = config.get("speaker_time_condition")
        if isinstance(speaker_time_condition_cfg, DictConfig):
            speaker_time_condition_cfg = OmegaConf.to_container(
                speaker_time_condition_cfg, resolve=True
            )
        detach_speaker_time_input = False
        speaker_time_condition = None
        if isinstance(speaker_time_condition_cfg, dict) and bool(
            speaker_time_condition_cfg.get("enabled", False)
        ):
            detach_speaker_time_input = bool(
                speaker_time_condition_cfg.get("detach_input", False)
            )
            speaker_time_input_dim = int(
                getattr(
                    getattr(speaker_encoder.speaker_encoder, "conv", None),
                    "out_channels",
                    512 * 3,
                )
            )
            speaker_time_condition = SpeakerTimeAdapter(
                input_dim=speaker_time_input_dim,
                hidden_dim=speaker_time_condition_cfg.get("hidden_dim"),
                num_layers=int(speaker_time_condition_cfg.get("num_layers", 2)),
                kernel_size=int(speaker_time_condition_cfg.get("kernel_size", 3)),
                dropout=float(speaker_time_condition_cfg.get("dropout", 0.0)),
            )

        return cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            speaker_adapter=speaker_adapter,
            prenet=prenet,
            postnet=postnet,
            speaker_time_condition=speaker_time_condition,
            speaker_input_key=speaker_input_key,
            detach_speaker_time_input=detach_speaker_time_input,
            use_codebook=config.get("use_codebook", True),
        )

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.
        
        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')
        model = cls.from_config(config)

        state_dict = load_file(ckpt_path)
        if any(key.startswith("speaker_conditioner.") for key in state_dict):
            state_dict = {
                (
                    key.replace("speaker_conditioner.", "speaker_adapter.", 1)
                    if key.startswith("speaker_conditioner.")
                    else key
                ): value
                for key, value in state_dict.items()
            }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def _select_speaker_wav(self, batch: Dict[str, Any]) -> torch.Tensor:
        for key in (self.speaker_input_key, "ref_wav", "wav"):
            value = batch.get(key)
            if value is not None:
                return value
        raise KeyError(
            f"BiCodec requires one of {self.speaker_input_key!r}, 'ref_wav', or 'wav' in batch"
        )

    def _target_length(self, semantic_tokens: torch.Tensor) -> int:
        if semantic_tokens.dim() == 2:
            return int(semantic_tokens.shape[-1])
        return int(semantic_tokens.shape[1])

    def _encode_speaker(
        self, speaker_wav: torch.Tensor, target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_transformer(speaker_wav).squeeze(1)
        if self.speaker_time_condition is None or target_length is None:
            return self.speaker_encoder(mel.transpose(1, 2))

        _, ecapa_latent = self.speaker_encoder.extract_xvector_and_ecapa_latent(
            mel.transpose(1, 2)
        )
        speaker_time_input = (
            ecapa_latent.detach() if self.detach_speaker_time_input else ecapa_latent
        )
        ecapa_latent = self.speaker_time_condition(
            speaker_time_input, target_length=target_length
        )
        return self.speaker_encoder.encode_from_ecapa_latent(ecapa_latent)

    def _tokenize_speaker(
        self, speaker_wav: torch.Tensor, target_length: Optional[int] = None
    ) -> torch.Tensor:
        mel = self.mel_transformer(speaker_wav).squeeze(1)
        if self.speaker_time_condition is None or target_length is None:
            return self.speaker_encoder.tokenize(mel.transpose(1, 2))

        _, ecapa_latent = self.speaker_encoder.extract_xvector_and_ecapa_latent(
            mel.transpose(1, 2)
        )
        speaker_time_input = (
            ecapa_latent.detach() if self.detach_speaker_time_input else ecapa_latent
        )
        ecapa_latent = self.speaker_time_condition(
            speaker_time_input, target_length=target_length
        )
        return self.speaker_encoder.tokenize_from_ecapa_latent(ecapa_latent)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.
        
        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        z = self.encoder(feat.transpose(1, 2))

        if self.use_codebook:
            vq_outputs = self.quantizer(z)
            z_q = vq_outputs["z_q"]
            vq_loss = vq_outputs["vq_loss"]
            perplexity = vq_outputs["perplexity"]
            active_num = vq_outputs["active_num"]
        else:
            z_q = z
            vq_loss = z.new_tensor(0.0)
            perplexity = z.new_tensor(0.0)
            active_num = z.new_tensor(0.0)

        speaker_wav = self._select_speaker_wav(batch)
        x_vector, d_vector = self._encode_speaker(
            speaker_wav, target_length=z_q.shape[-1]
        )

        use_detached = getattr(self, "_bicodec_detach_xvector", False)
        conditions = x_vector.detach() if use_detached else x_vector
        if self.speaker_adapter is not None:
            conditions = self.speaker_adapter(conditions)
        
        with_speaker_loss = False
        x = self.prenet(z_q, conditions)
        pred_feat = self.postnet(x) # pred wav2vec features
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        out: Dict[str, Any] = {
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "cluster_size": active_num,
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "condition_vector": conditions,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }
        clf = getattr(self, "speaker_classifier", None)
        if clf is not None:
            clf_input = conditions if self.speaker_adapter is not None else x_vector.detach()
            out["speaker_logits"] = clf(clf_input)
        return out

    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        speaker_wav = self._select_speaker_wav(batch)
        z = self.encoder(feat.transpose(1, 2))
        if self.use_codebook:
            semantic_tokens = self.quantizer.tokenize(z)
        else:
            semantic_tokens = z.transpose(1, 2)
        global_tokens = self._tokenize_speaker(
            speaker_wav, target_length=self._target_length(semantic_tokens)
        )

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        if self.use_codebook:
            z_q = self.quantizer.detokenize(semantic_tokens)
        else:
            z_q = semantic_tokens
            if z_q.dim() == 3 and z_q.shape[-1] == self.quantizer.input_dim:
                z_q = z_q.transpose(1, 2)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


# Test the model
if __name__ == "__main__":

    config = load_config("pretrained_models/SparkTTS-0.5B/BiCodec/config.yaml")
    model = BiCodec.load_from_checkpoint(
        model_dir="pretrained_models/SparkTTS-0.5B/BiCodec",
    )

    # Generate random inputs for testing
    duration = 0.96
    x = torch.randn(20, 1, int(duration * 16000))
    feat = torch.randn(20, int(duration * 50), 1024)
    inputs = {"feat": feat, "wav": x, "ref_wav": x}

    # Forward pass
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)

    # Verify if the reconstruction matches
    if torch.allclose(outputs["recons"].detach(), wav_recon):
        print("Test successful")
    else:
        print("Test failed")
