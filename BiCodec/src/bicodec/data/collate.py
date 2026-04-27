from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F


def pad_1d_batch(items: Sequence[torch.Tensor]) -> torch.Tensor:
    max_len = max(item.shape[-1] for item in items)
    padded = []
    for item in items:
        if item.shape[-1] == max_len:
            padded.append(item)
            continue
        padded.append(F.pad(item, (0, max_len - item.shape[-1])))
    return torch.stack(padded, dim=0)


def collate_audio_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "wav": pad_1d_batch([item["wav"] for item in batch]),
        "ref_wav": pad_1d_batch([item["ref_wav"] for item in batch]),
        "audio_filepath": [item["audio_filepath"] for item in batch],
    }
    if batch and all("speaker_label" in item for item in batch):
        output["speaker_label"] = torch.tensor(
            [int(item["speaker_label"]) for item in batch],
            dtype=torch.long,
        )
    return output
