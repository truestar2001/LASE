import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from bicodec.data.manifest import load_manifest
from bicodec.utils.audio import load_audio


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
        speaker_id_key: str = "speaker_id",
        speaker_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.entries = load_manifest(self.manifest_path)
        self.sample_rate = int(sample_rate)
        self.audio_key = audio_key
        self.base_dir = Path(base_dir) if base_dir else None
        self.volume_normalize = volume_normalize
        self.segment_duration = segment_duration
        self.ref_segment_duration = ref_segment_duration
        self.speaker_id_key = speaker_id_key
        self.speaker_id_to_idx = speaker_id_to_idx
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
        audio_path = Path(entry[self.audio_key])
        if audio_path.is_absolute() or self.base_dir is None:
            return audio_path
        return self.base_dir / audio_path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        audio_path = self._resolve_audio_path(entry)
        wav = load_audio(
            audio_path,
            sampling_rate=self.sample_rate,
            volume_normalize=self.volume_normalize,
            segment_duration=self.segment_duration,
        )
        wav_tensor = torch.from_numpy(wav).float()
        output: Dict[str, Any] = {
            "wav": wav_tensor,
            "ref_wav": build_reference_clip(
                wav_tensor,
                sample_rate=self.sample_rate,
                duration=self.ref_segment_duration,
            ),
            "audio_filepath": str(audio_path),
        }
        if self.speaker_id_to_idx is not None:
            speaker_id = entry.get(self.speaker_id_key)
            if speaker_id is None:
                raise KeyError(
                    f"Missing {self.speaker_id_key!r} for {audio_path} "
                    "(required when using speaker_id_to_idx)"
                )
            output["speaker_label"] = self.speaker_id_to_idx[str(speaker_id)]
        return output
