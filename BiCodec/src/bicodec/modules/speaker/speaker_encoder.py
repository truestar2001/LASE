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

from typing import List, Tuple
from bicodec.modules.fsq.residual_fsq import ResidualFSQ
from bicodec.modules.speaker.ecapa_tdnn import ECAPA_TDNN_GLOB_c512
from bicodec.modules.speaker.perceiver_encoder import PerceiverResampler

"""
x-vector + d-vector
"""


class SpeakerEncoder(nn.Module):
    """

    Args:
        input_dim (int): acoustic feature dimension
        out_dim (int): output dimension of x-vector and d-vector
        latent_dim (int): latent dimension before quantization
        token_num (int): sequence length of speaker tokens
        fsq_levels (List[int]): number of levels for each quantizer
        fsq_num_quantizers (int): number of quantizers

    Return:
        speaker_embs: (B, T2, out_dim)
    """

    def __init__(
        self,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: List[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
    ):
        super(SpeakerEncoder, self).__init__()

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(
            feat_dim=input_dim, embed_dim=out_dim
        )
        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim, dim_context=512 * 3, num_latents=token_num
        )
        self.quantizer = ResidualFSQ(
            levels=fsq_levels,
            num_quantizers=fsq_num_quantizers,
            dim=latent_dim,
            is_channel_first=True,
            quantize_dropout=False,
        )

        self.project = nn.Linear(latent_dim * token_num, out_dim)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1, 2))
        return zq.transpose(1, 2)

    def get_indices(self, mels: torch.Tensor) -> torch.Tensor:
        mels = mels.transpose(1, 2)
        x = self.perceiver_sampler(mels).transpose(1, 2)
        zq, indices = self.quantizer(x)
        return indices

    def extract_xvector_and_ecapa_latent(
        self, mels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.speaker_encoder(mels, True)

    def _quantize_from_ecapa_latent(
        self, ecapa_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.perceiver_sampler(ecapa_latent.transpose(1, 2)).transpose(1, 2)
        return self.quantizer(x)

    def _project_d_vector(self, zq: torch.Tensor) -> torch.Tensor:
        x = zq.reshape(zq.shape[0], -1)
        return self.project(x)

    def encode_from_ecapa_latent(
        self, ecapa_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_vector = self.speaker_encoder.project_latent(ecapa_latent)
        zq, _ = self._quantize_from_ecapa_latent(ecapa_latent)
        d_vector = self._project_d_vector(zq)
        return x_vector, d_vector

    def forward(
        self, mels: torch.Tensor, return_ecapa_latent: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            mels: (B, D_mel, T1)

        Return:
            x_vector: (B, out_dim)
            d_vector: (B, out_dim)
        """
        x_vector, ecapa_latent = self.extract_xvector_and_ecapa_latent(mels)
        zq, _ = self._quantize_from_ecapa_latent(ecapa_latent)
        d_vector = self._project_d_vector(zq)

        if return_ecapa_latent:
            return x_vector, d_vector, ecapa_latent
        return x_vector, d_vector

    def tokenize_from_ecapa_latent(self, ecapa_latent: torch.Tensor) -> torch.Tensor:
        """tokenize precomputed ECAPA latent"""
        _, indices = self._quantize_from_ecapa_latent(ecapa_latent)
        return indices

    def tokenize(self, mels: torch.Tensor) -> torch.Tensor:
        """tokenize the input mel spectrogram"""
        _, ecapa_latent = self.extract_xvector_and_ecapa_latent(mels)
        return self.tokenize_from_ecapa_latent(ecapa_latent)

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """detokenize the input indices to d-vector"""
        zq = self.quantizer.get_output_from_indices(indices.transpose(1, 2)).transpose(1, 2)
        return self._project_d_vector(zq)

if __name__ == "__main__":
    model = SpeakerEncoder(
        input_dim=100,
        latent_dim=128,
        token_num=32,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=1,
    )
    mel = torch.randn(8, 200, 100)
    x_vector, d_vector = model(mel)
    print("x-vector shape", x_vector.shape)
    print("d-vector shape", d_vector.shape)

    indices = model.tokenize(mel)
    print("indices shape", indices.shape)
    d_vector_post = model.detokenize(indices)
    print("d-vector shape", d_vector_post.shape)
    if d_vector_post.all() == d_vector.all():
        print("d-vector post and d-vector are the same")
    else:
        print("d-vector post and d-vector are different")
    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
