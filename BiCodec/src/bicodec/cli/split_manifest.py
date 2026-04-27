import argparse
from collections import Counter
from pathlib import Path

from bicodec.data.manifest import load_manifest
from bicodec.data.split import split_manifest_by_speaker
from bicodec.utils.jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a speaker-stratified train/valid split from a manifest"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--speaker-id-key", type=str, default="speaker_id")
    parser.add_argument("--valid-per-speaker", type=int, default=3)
    parser.add_argument("--min-train-per-speaker", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _speaker_summary(rows, speaker_id_key: str) -> str:
    counts = Counter(str(row[speaker_id_key]) for row in rows)
    return (
        f"speakers={len(counts)} "
        f"samples={sum(counts.values())} "
        f"min_per_speaker={min(counts.values()) if counts else 0} "
        f"max_per_speaker={max(counts.values()) if counts else 0}"
    )


def main() -> None:
    args = parse_args()
    entries = load_manifest(args.input_manifest)
    train_rows, valid_rows = split_manifest_by_speaker(
        entries,
        speaker_id_key=args.speaker_id_key,
        valid_per_speaker=args.valid_per_speaker,
        min_train_per_speaker=args.min_train_per_speaker,
        seed=args.seed,
    )

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.valid_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, args.train_output)
    write_jsonl(valid_rows, args.valid_output)

    print(f"[split] input={args.input_manifest}")
    print(f"[split] train={args.train_output} {_speaker_summary(train_rows, args.speaker_id_key)}")
    print(f"[split] valid={args.valid_output} {_speaker_summary(valid_rows, args.speaker_id_key)}")


if __name__ == "__main__":
    main()
