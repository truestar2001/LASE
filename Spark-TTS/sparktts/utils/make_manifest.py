import json
from pathlib import Path

# VoxCeleb1 루트 경로
# DATA_ROOT = Path("/shared/NAS_HDD/yoon/lase/voxceleb1").resolve()
DATA_ROOT = Path("/shared/NAS_HDD/yoon/lase/voxceleb2").resolve()


# 저장할 manifest 파일
OUTPUT_PATH = Path("/home/jyp/LASE/Spark-TTS/manifests/voxceleb2.json").resolve()

entries = []

for wav_path in DATA_ROOT.rglob("*.wav"):
    # 예: /.../data/id10001/1zcIwhmdeo4/00001.wav
    # speaker_id = id10001
    speaker_id = wav_path.relative_to(DATA_ROOT).parts[0]

    entries.append({
        "audio_filepath": str(wav_path.resolve()),
        "speaker_id": speaker_id
    })

# 정렬하고 저장
entries.sort(key=lambda x: x["audio_filepath"])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in entries:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(entries)} entries to {OUTPUT_PATH}")