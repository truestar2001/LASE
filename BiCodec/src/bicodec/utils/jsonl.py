import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().splitlines() if line.strip()]


def write_jsonl(rows: Iterable[Dict[str, Any]], file_path: Path) -> None:
    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
