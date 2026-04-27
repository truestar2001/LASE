import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bicodec.data.manifest import build_speaker_label_map, load_manifest


def test_load_manifest_jsonl(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"audio_filepath": "a.wav", "speaker_id": "spk1"}),
                json.dumps({"audio_filepath": "b.wav", "speaker_id": "spk2"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_manifest(manifest_path)
    assert len(rows) == 2
    assert rows[0]["audio_filepath"] == "a.wav"


def test_build_speaker_label_map_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(
        json.dumps(
            {
                "0": {"audio_filepath": "a.wav", "speaker_id": "spk_b"},
                "1": {"audio_filepath": "b.wav", "speaker_id": "spk_a"},
            }
        ),
        encoding="utf-8",
    )

    label_map, num_classes = build_speaker_label_map([str(manifest_path)])
    assert num_classes == 2
    assert label_map == {"spk_a": 0, "spk_b": 1}
