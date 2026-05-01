#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


# =========================
# User settings
# =========================
BASELINES = [
    "ecapa-tdnn",
    "ecapa2",
    "redimnet",
]

TRIAL_LIST = "/shared/NAS_HDD/yoon/lase/voxceleb1/trial_list/voxsrc_2021_val_trial_list.txt"
EMB_ROOT = "/shared/NAS_HDD/yoon/lase/embeddings"
OUT_CSV = "/home/jyp/LASE/evaluation/results/vox1b_results.csv"

P_TARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def parse_trials(trial_list_path: str):
    trials = []

    with open(trial_list_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed line at {line_num}: {line}")

            label_str, rel1, rel2 = parts

            if label_str not in {"0", "1"}:
                raise ValueError(f"Invalid label at {line_num}: {line}")

            if not rel1.endswith(".wav") or not rel2.endswith(".wav"):
                raise ValueError(f"Invalid wav path at {line_num}: {line}")

            trials.append((int(label_str), rel1, rel2))

    return trials


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / max(np.linalg.norm(x), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    c_det = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = np.min(c_det) / c_def
    return float(min_dcf)


def rel_wav_to_emb_path(emb_root: Path, baseline: str, rel_wav_path: str) -> Path:
    return emb_root / baseline / Path('voxceleb1/data') / Path(rel_wav_path).with_suffix(".npy")


def load_embedding(path: Path) -> np.ndarray:
    emb = np.load(str(path))
    emb = np.asarray(emb, dtype=np.float32).squeeze()

    if emb.ndim != 1:
        raise ValueError(f"Embedding shape must be 1D after squeeze, but got {emb.shape} from {path}")

    return emb


def evaluate_one_baseline(baseline: str, trials, emb_root: Path):
    print(f"[INFO] Evaluating: {baseline}")

    cache = {}
    labels = []
    scores = []

    for idx, (label, rel1, rel2) in enumerate(trials, 1):
        emb1_path = rel_wav_to_emb_path(emb_root, baseline, rel1)
        emb2_path = rel_wav_to_emb_path(emb_root, baseline, rel2)

        if not emb1_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb1_path}")
        if not emb2_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb2_path}")

        if rel1 not in cache:
            cache[rel1] = load_embedding(emb1_path)
        if rel2 not in cache:
            cache[rel2] = load_embedding(emb2_path)

        score = cosine_similarity(cache[rel1], cache[rel2])

        labels.append(label)
        scores.append(score)

        if idx % 50000 == 0:
            print(f"[INFO] {baseline}: processed {idx}/{len(trials)} trials")

    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    eer = compute_eer(labels, scores)
    min_dcf = compute_min_dcf(
        labels,
        scores,
        p_target=P_TARGET,
        c_miss=C_MISS,
        c_fa=C_FA,
    )

    return {
        "model": baseline,
        "eer": eer,
        "eer_percent": eer * 100.0,
        "minDCF": min_dcf,
    }


def main():
    emb_root = Path(EMB_ROOT)
    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Trial list : {TRIAL_LIST}")
    print(f"[INFO] Emb root   : {EMB_ROOT}")
    print(f"[INFO] Models     : {BASELINES}")
    print(f"[INFO] Output CSV : {OUT_CSV}")

    trials = parse_trials(TRIAL_LIST)
    print(f"[INFO] Num trials : {len(trials)}")

    rows = []
    for baseline in BASELINES:
        row = evaluate_one_baseline(baseline, trials, emb_root)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["model", "eer", "eer_percent", "minDCF"])
    df.to_csv(out_csv, index=False)

    print("\n[RESULT]")
    print(df.to_string(index=False))
    print(f"\n[INFO] Saved CSV to: {out_csv}")


if __name__ == "__main__":
    main()