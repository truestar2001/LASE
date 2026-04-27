from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bicodec.data.split import split_manifest_by_speaker


def test_split_manifest_by_speaker_keeps_train_and_valid_balanced() -> None:
    entries = []
    for speaker_id in ("spk_a", "spk_b"):
        for index in range(5):
            entries.append(
                {
                    "audio_filepath": f"{speaker_id}_{index}.wav",
                    "speaker_id": speaker_id,
                }
            )

    train_rows, valid_rows = split_manifest_by_speaker(
        entries,
        valid_per_speaker=2,
        min_train_per_speaker=1,
        seed=42,
    )

    assert len(train_rows) == 6
    assert len(valid_rows) == 4
    assert sum(row["speaker_id"] == "spk_a" for row in valid_rows) == 2
    assert sum(row["speaker_id"] == "spk_b" for row in valid_rows) == 2
