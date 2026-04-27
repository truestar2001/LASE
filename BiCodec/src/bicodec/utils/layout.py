from pathlib import Path
from typing import Dict


def resolve_artifact_layout(root: Path) -> Dict[str, Path]:
    root = Path(root)

    bicodec_dir = None
    for candidate in (root / "bicodec", root / "BiCodec", root):
        if (candidate / "config.yaml").exists():
            bicodec_dir = candidate
            break

    if bicodec_dir is None:
        bicodec_dir = root / "bicodec"

    return {
        "artifact_root": root,
        "bicodec_dir": bicodec_dir,
        "model_config_path": bicodec_dir / "config.yaml",
        "model_checkpoint_path": bicodec_dir / "model.safetensors",
        "feature_extractor_dir": root / "wav2vec2-large-xlsr-53",
    }
