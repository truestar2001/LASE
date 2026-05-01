#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


VOX1_ROOT = "/shared/NAS_HDD/yoon/lase/voxceleb1"
VOX1_CD_ROOT = "/shared/NAS_HDD/yoon/lase/voxceleb1_cd"
EMB_ROOT = "/shared/NAS_HDD/yoon/lase/embeddings"
BASELINE_NAME = "redimnet"


def collect_all_wavs(dataset_name: str, dataset_root: Path) -> List[Tuple[str, Path, Path]]:
    """
    Return list of:
      (dataset_name, absolute_wav_path, relative_path_from_dataset_root)

    Example:
      ("voxceleb1", /.../voxceleb1/id10001/xxx.wav, id10001/xxx.wav)
    """
    items: List[Tuple[str, Path, Path]] = []

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    for wav_path in sorted(dataset_root.rglob("*.wav")):
        if not wav_path.is_file():
            continue
        rel_path = wav_path.relative_to(dataset_root)
        items.append((dataset_name, wav_path, rel_path))

    return items


def load_audio_16k_mono(wav_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)  # [T]


def save_embedding(out_path: Path, emb: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), emb.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vox1-root", type=str, default=VOX1_ROOT)
    parser.add_argument("--vox1-cd-root", type=str, default=VOX1_CD_ROOT)
    parser.add_argument("--emb-root", type=str, default=EMB_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .npy already exists.")
    parser.add_argument("--model-name", type=str, default="M")
    parser.add_argument("--train-type", type=str, default="ft_mix")
    parser.add_argument("--dataset", type=str, default="vb2+vox2+cnc")
    parser.add_argument("--repo", type=str, default="IDRnD/ReDimNet")
    args = parser.parse_args()

    vox1_root = Path(args.vox1_root)
    vox1_cd_root = Path(args.vox1_cd_root)
    baseline_root = Path(args.emb_root) / BASELINE_NAME
    baseline_root.mkdir(parents=True, exist_ok=True)

    all_items: List[Tuple[str, Path, Path]] = []
    all_items.extend(collect_all_wavs("voxceleb1", vox1_root))
    all_items.extend(collect_all_wavs("voxceleb1_cd", vox1_cd_root))

    num_vox1 = sum(1 for dataset_name, _, _ in all_items if dataset_name == "voxceleb1")
    num_vox1_cd = sum(1 for dataset_name, _, _ in all_items if dataset_name == "voxceleb1_cd")

    print(f"[INFO] baseline         : {BASELINE_NAME}")
    print(f"[INFO] vox1 root        : {vox1_root}")
    print(f"[INFO] vox1_cd root     : {vox1_cd_root}")
    print(f"[INFO] emb root         : {baseline_root}")
    print(f"[INFO] device           : {args.device}")
    print(f"[INFO] num vox1 wavs    : {num_vox1}")
    print(f"[INFO] num vox1_cd wavs : {num_vox1_cd}")
    print(f"[INFO] total wavs       : {len(all_items)}")

    model = torch.hub.load(
        args.repo,
        'ReDimNet',
        model_name='b2',
        train_type='ft_lm',
        dataset='vox2',
    )
    model = model.to(args.device)
    model.eval()

    device_type = "cuda" if "cuda" in args.device and torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if device_type == "cuda" else torch.float32

    done = 0
    skipped = 0
    missing = []

    per_dataset_done = {"voxceleb1": 0, "voxceleb1_cd": 0}
    per_dataset_skipped = {"voxceleb1": 0, "voxceleb1_cd": 0}
    per_dataset_missing = {"voxceleb1": 0, "voxceleb1_cd": 0}

    for dataset_name, wav_path, rel_path in tqdm(all_items, desc=f"Extracting [{BASELINE_NAME}]"):
        out_path = baseline_root / dataset_name / rel_path.with_suffix(".npy")

        if out_path.exists() and not args.force:
            skipped += 1
            per_dataset_skipped[dataset_name] += 1
            continue

        if not wav_path.exists():
            missing.append(str(wav_path))
            per_dataset_missing[dataset_name] += 1
            continue

        wav = load_audio_16k_mono(wav_path).unsqueeze(0).to(args.device)  # [1, T]

        with torch.no_grad():
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    emb = model(wav)
            else:
                emb = model(wav)

            emb = emb.squeeze().detach().cpu().numpy()

        save_embedding(out_path, emb)
        done += 1
        per_dataset_done[dataset_name] += 1

    summary = {
        "baseline": BASELINE_NAME,
        "num_voxceleb1_wavs": num_vox1,
        "num_voxceleb1_cd_wavs": num_vox1_cd,
        "num_total_wavs": len(all_items),
        "done": done,
        "skipped": skipped,
        "missing": missing,
        "model_name": args.model_name,
        "train_type": args.train_type,
        "dataset": args.dataset,
        "repo": args.repo,
        "per_dataset_done": per_dataset_done,
        "per_dataset_skipped": per_dataset_skipped,
        "per_dataset_missing": per_dataset_missing,
    }

    with open(baseline_root / "extract_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()