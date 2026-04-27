from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bicodec.utils.config import load_config


def test_exp_configs_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "train_bicodec_exp1.yaml",
        "train_bicodec_exp2.yaml",
        "train_bicodec_exp3.yaml",
    ):
        cfg = load_config(repo_root / "configs" / name)
        assert cfg.model is not None
        assert cfg.trainer is not None
