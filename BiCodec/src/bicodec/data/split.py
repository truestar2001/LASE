import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def split_manifest_by_speaker(
    entries: Sequence[Dict[str, Any]],
    *,
    speaker_id_key: str = "speaker_id",
    valid_per_speaker: int = 3,
    min_train_per_speaker: int = 1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if valid_per_speaker < 0:
        raise ValueError("valid_per_speaker must be >= 0")
    if min_train_per_speaker < 1:
        raise ValueError("min_train_per_speaker must be >= 1")

    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        speaker_id = entry.get(speaker_id_key)
        if speaker_id is None:
            raise KeyError(f"Missing {speaker_id_key!r} in manifest entry at index {index}")
        grouped[str(speaker_id)].append((index, entry))

    rng = random.Random(seed)
    train_rows: List[Tuple[int, Dict[str, Any]]] = []
    valid_rows: List[Tuple[int, Dict[str, Any]]] = []

    for speaker_id in sorted(grouped):
        items = list(grouped[speaker_id])
        rng.shuffle(items)

        max_valid = max(0, len(items) - min_train_per_speaker)
        valid_count = min(valid_per_speaker, max_valid)

        valid_rows.extend(items[:valid_count])
        train_rows.extend(items[valid_count:])

    train_rows.sort(key=lambda item: item[0])
    valid_rows.sort(key=lambda item: item[0])
    return [entry for _, entry in train_rows], [entry for _, entry in valid_rows]
