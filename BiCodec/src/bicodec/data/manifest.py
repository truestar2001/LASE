import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from bicodec.utils.jsonl import read_jsonl


def _read_json_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.values())

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL manifest at {path}, line {lineno}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object per line in {path}, line {lineno}, got {type(row)}"
                )
            rows.append(row)
    return rows


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        return _read_json_manifest(path)
    raise ValueError(f"Unsupported manifest extension: {path}")


def build_speaker_label_map(
    manifest_paths: Union[str, Sequence[str]],
    speaker_id_key: str = "speaker_id",
) -> Tuple[Dict[str, int], int]:
    paths = [manifest_paths] if isinstance(manifest_paths, str) else list(manifest_paths)
    speaker_ids = set()
    for manifest_path in paths:
        if not manifest_path:
            continue
        for entry in load_manifest(Path(manifest_path)):
            speaker_id = entry.get(speaker_id_key)
            if speaker_id is not None:
                speaker_ids.add(str(speaker_id))

    ordered_ids = sorted(speaker_ids)
    if not ordered_ids:
        raise ValueError(
            f"No {speaker_id_key!r} in manifests {paths} (need labels for speaker classification)."
        )
    return {speaker_id: idx for idx, speaker_id in enumerate(ordered_ids)}, len(ordered_ids)
