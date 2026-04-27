from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Path) -> DictConfig:
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)

    base_config = config.get("base_config")
    if base_config is None:
        return config

    base_config_path = Path(base_config)
    if not base_config_path.is_absolute():
        base_config_path = config_path.parent / base_config_path
    return OmegaConf.merge(OmegaConf.load(base_config_path), config)
