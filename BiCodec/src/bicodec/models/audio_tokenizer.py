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
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from bicodec.utils.config import load_config
from bicodec.utils.layout import resolve_artifact_layout
from bicodec.utils.audio import load_audio
from bicodec.models.bicodec import BiCodec


class BiCodecTokenizer:
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device = None,
        use_codebook: bool = None,
        **kwargs,
    ):
        super().__init__()
        """
        Args:
            model_dir: Path to an artifact root containing `bicodec/` and `wav2vec2-large-xlsr-53/`.
            device: Device to run the model on (default is GPU if available).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.layout = resolve_artifact_layout(self.model_dir)
        self.use_codebook_override = use_codebook
        self.config = load_config(self.layout["model_config_path"])
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = BiCodec.load_from_checkpoint(self.layout["bicodec_dir"]).to(self.device)
        if self.use_codebook_override is not None:
            self.model.use_codebook = bool(self.use_codebook_override)
        self.use_codebook = self.model.use_codebook
        feature_extractor_dir = self.layout["feature_extractor_dir"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_dir)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(feature_extractor_dir).to(
            self.device
        )
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """load auido and get reference audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav)

        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()
        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens:
                - codebook on: token ids
                - codebook off: continuous semantic latents
            global_tokens: global tokens
        """
        wavs = batch["wav"]
        feats = self.extract_wav2vec2_features(wavs)
        if isinstance(wavs, (list, tuple)):
            wav_tensors = []
            for wav in wavs:
                if torch.is_tensor(wav):
                    wav_tensors.append(wav.float())
                else:
                    wav_tensors.append(torch.from_numpy(np.asarray(wav)).float())
            batch["wav"] = torch.nn.utils.rnn.pad_sequence(
                wav_tensors, batch_first=True
            ).to(self.device)
        elif torch.is_tensor(wavs):
            batch["wav"] = wavs.to(self.device)
        batch["feat"] = feats
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        wav, ref_wav = self.process_audio(audio_path)
        feat = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": ref_wav.to(self.device),
            "feat": feat.to(self.device),
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens
            semantic_tokens:
                - codebook on: token ids
                - codebook off: continuous semantic latents

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        global_tokens = global_tokens.unsqueeze(1)
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        return wav_rec.detach().squeeze().cpu().numpy()


# test
if __name__ == "__main__":
    import soundfile as sf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(
        model_dir="artifacts/pretrained/bicodec-base",
        device=device,
    )
    wav_path = "example/prompt_audio.wav"

    global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)

    wav_rec = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)
    sf.write("example/prompt_recon.wav", wav_rec, 16000)
