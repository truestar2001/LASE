"""Training-time setup utilities.

This module is organized in the order the training stack is usually read:
1. runtime setup
2. freeze policy
3. model setup
4. checkpoint IO
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from bicodec.data.manifest import build_speaker_label_map
from bicodec.models.bicodec import BiCodec
from bicodec.utils.config import load_config
from bicodec.utils.layout import resolve_artifact_layout


BICODEC_FREEZABLE: Tuple[str, ...] = (
    "encoder",
    "quantizer",
    "speaker_encoder",
    "speaker_adapter",
    "speaker_time_condition",
    "prenet",
    "postnet",
    "decoder",
    "mel_transformer",
    "speaker_classifier",
)

FREEZE_NAME_ALIASES: Dict[str, str] = {
    "speaker_conditioner": "speaker_adapter",
    "speaker_time_adapter": "speaker_time_condition",
}


# Runtime setup


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_device_and_ddp(cfg: DictConfig) -> Tuple[torch.device, int, bool]:
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return device, local_rank, True

    device_name = cfg.trainer.get("device")
    if device_name:
        return torch.device(device_name), 0, False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, False


def cleanup_ddp() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def ddp_average_scalar(value: float, device: torch.device) -> float:
    if not is_dist_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())


def ddp_average_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_dist_initialized():
        return metrics
    return {key: ddp_average_scalar(float(value), device) for key, value in metrics.items()}


class FrozenWav2Vec2Frontend(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        hidden_state_indices: Sequence[int],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.model = Wav2Vec2Model.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_state_indices = tuple(hidden_state_indices)

    @torch.no_grad()
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            [wav.cpu().numpy() for wav in wavs],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        outputs = self.model(
            inputs.input_values.to(self.model.device),
            output_hidden_states=True,
        )
        hidden_states = [outputs.hidden_states[idx] for idx in self.hidden_state_indices]
        return sum(hidden_states) / len(hidden_states)


# Freeze policy


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def effective_trainer_freeze(cfg: DictConfig) -> Optional[Dict[str, bool]]:
    raw_freeze = cfg.trainer.get("freeze")
    if raw_freeze is not None:
        freeze_dict = OmegaConf.to_container(raw_freeze, resolve=True)
        if isinstance(freeze_dict, dict):
            normalized: Dict[str, bool] = {}
            for name, frozen in freeze_dict.items():
                name = FREEZE_NAME_ALIASES.get(str(name), str(name))
                if name not in BICODEC_FREEZABLE or not isinstance(frozen, (bool, int)):
                    continue
                normalized[name] = bool(frozen)
            if normalized:
                return normalized

    if bool(cfg.trainer.get("freeze_semantic_encoder", False)):
        return {"encoder": True, "quantizer": True}
    return None


def apply_trainer_freeze_from_config(model: nn.Module, cfg: DictConfig) -> None:
    freeze_spec = effective_trainer_freeze(cfg)
    if not freeze_spec:
        return

    module = unwrap_model(model)
    for name, frozen in freeze_spec.items():
        submodule = getattr(module, name, None)
        if submodule is None:
            continue
        submodule.requires_grad_(not frozen)


def set_bicodec_experiment_model_train_mode(model: nn.Module, cfg: DictConfig) -> None:
    module = unwrap_model(model)
    module.train()

    freeze_spec = effective_trainer_freeze(cfg)
    if not freeze_spec:
        return

    for name, frozen in freeze_spec.items():
        submodule = getattr(module, name, None)
        if submodule is None:
            continue
        if frozen:
            submodule.eval()
        else:
            submodule.train()


# Model setup


def speaker_xvector_dim(model_config: Any) -> int:
    model_config = OmegaConf.to_container(model_config, resolve=True)
    if not isinstance(model_config, dict):
        return 512
    if "audio_tokenizer" in model_config and isinstance(model_config["audio_tokenizer"], dict):
        model_config = model_config["audio_tokenizer"]
    speaker_encoder_cfg = model_config.get("speaker_encoder", {})
    if isinstance(speaker_encoder_cfg, dict) and "out_dim" in speaker_encoder_cfg:
        return int(speaker_encoder_cfg["out_dim"])
    return 512


def initialize_training_model(
    cfg: DictConfig,
    model: nn.Module,
    model_config: Any,
    device: torch.device,
    *,
    do_print: bool = True,
) -> Optional[Dict[str, int]]:
    module = unwrap_model(model)
    speaker_id_to_idx: Optional[Dict[str, int]] = None

    if bool(cfg.trainer.get("speaker_classification", False)):
        manifest_paths = [cfg.data.train_manifest]
        if cfg.data.get("valid_manifest"):
            manifest_paths.append(cfg.data.valid_manifest)
        speaker_id_key = str(cfg.data.get("speaker_id_key", "speaker_id"))
        speaker_id_to_idx, num_classes = build_speaker_label_map(manifest_paths, speaker_id_key)
        module.add_module(
            "speaker_classifier",
            nn.Linear(speaker_xvector_dim(model_config), num_classes).to(device),
        )
        if do_print:
            print(
                f"[train] speaker_classification  classes={num_classes}  "
                f"x_vector_dim={speaker_xvector_dim(model_config)}"
            )

    apply_trainer_freeze_from_config(model, cfg)
    module._bicodec_detach_xvector = bool(cfg.trainer.get("detach_xvector_condition", False))

    if do_print and getattr(module, "speaker_time_condition", None) is not None:
        print(
            "[train] speaker_time_condition  "
            f"speaker_input_key={getattr(module, 'speaker_input_key', 'ref_wav')}  "
            f"detach_input={bool(getattr(module, 'detach_speaker_time_input', False))}"
        )
    return speaker_id_to_idx


def apply_bicodec_model_overrides(
    model_config: Any,
    *,
    speaker_input_key: Optional[str] = None,
    speaker_time_condition: Optional[Any] = None,
) -> DictConfig:
    config_dict = OmegaConf.to_container(model_config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError(f"Expected dict-like model config, got {type(config_dict)}")

    target = config_dict.get("audio_tokenizer", config_dict)
    if not isinstance(target, dict):
        raise TypeError(f"Expected dict-like audio_tokenizer config, got {type(target)}")

    if speaker_input_key is not None:
        target["speaker_input_key"] = str(speaker_input_key)

    if speaker_time_condition is not None:
        if isinstance(speaker_time_condition, DictConfig):
            speaker_time_condition = OmegaConf.to_container(
                speaker_time_condition,
                resolve=True,
            )
        if isinstance(speaker_time_condition, dict):
            target["speaker_time_adapter"] = speaker_time_condition

    return OmegaConf.create(config_dict)


def load_state_dict_for_training(path: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".safetensors":
        return load_safetensors(str(checkpoint_path))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_bicodec_model(
    config_path: str,
    init_ckpt_path: Optional[str],
    strict_load: bool,
    quantizer_distance_loss_type: Optional[str] = None,
    use_codebook: Optional[bool] = None,
    speaker_input_key: Optional[str] = None,
    speaker_time_condition: Optional[Any] = None,
) -> Tuple[BiCodec, DictConfig]:
    model_config = apply_bicodec_model_overrides(
        load_config(config_path),
        speaker_input_key=speaker_input_key,
        speaker_time_condition=speaker_time_condition,
    )
    model = BiCodec.from_config(model_config)

    if use_codebook is not None:
        model.use_codebook = bool(use_codebook)
    if quantizer_distance_loss_type:
        model.quantizer.distance_loss_type = quantizer_distance_loss_type

    if init_ckpt_path:
        state_dict = load_state_dict_for_training(init_ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_load)
        if missing_keys and is_main_process():
            print(f"[init] missing keys: {missing_keys}")
        if unexpected_keys and is_main_process():
            print(f"[init] unexpected keys: {unexpected_keys}")

    return model, model_config


def resolve_training_paths(cfg: DictConfig) -> Dict[str, str]:
    pretrained_model_dir = cfg.model.get("pretrained_model_dir")
    model_config_path = cfg.model.get("config_path")
    init_ckpt_path = cfg.model.get("init_ckpt_path")
    feature_extractor_path = cfg.model.get("feature_extractor_path")

    if pretrained_model_dir is not None:
        layout = resolve_artifact_layout(Path(pretrained_model_dir))
        model_config_path = model_config_path or str(layout["model_config_path"])
        init_ckpt_path = init_ckpt_path or str(layout["model_checkpoint_path"])
        feature_extractor_path = feature_extractor_path or str(layout["feature_extractor_dir"])

    if model_config_path is None:
        raise ValueError("`model.config_path` or `model.pretrained_model_dir` is required")
    if feature_extractor_path is None:
        raise ValueError("`model.feature_extractor_path` or `model.pretrained_model_dir` is required")

    return {
        "model_config_path": model_config_path,
        "init_ckpt_path": init_ckpt_path,
        "feature_extractor_path": feature_extractor_path,
    }


def make_optimizer(cfg: DictConfig, params) -> AdamW:
    return AdamW(
        params,
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.optim.get("weight_decay", 0.0)),
    )


# Checkpoint IO


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler_g: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: Optional[float],
    cfg: DictConfig,
    discriminator_state: Optional[Dict[str, Any]] = None,
) -> None:
    if not is_main_process():
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer_g.state_dict(),
        "scheduler": None if scheduler_g is None else scheduler_g.state_dict(),
        "scaler": scaler_g.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if discriminator_state is not None:
        payload["discriminator"] = discriminator_state
    torch.save(payload, checkpoint_path)


def maybe_load_resume(
    resume_path: Optional[str],
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler_g: GradScaler,
    mpd: Optional[nn.Module] = None,
    mrd: Optional[nn.Module] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler_d: Optional[GradScaler] = None,
) -> Tuple[int, int, Optional[float]]:
    if not resume_path:
        return 0, 0, None

    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(payload["state_dict"])
    optimizer_g.load_state_dict(payload["optimizer"])

    if scheduler_g is not None and payload.get("scheduler") is not None:
        scheduler_g.load_state_dict(payload["scheduler"])
    if payload.get("scaler") is not None:
        scaler_g.load_state_dict(payload["scaler"])

    discriminator_state = payload.get("discriminator")
    if discriminator_state is not None:
        if mpd is not None and discriminator_state.get("mpd") is not None:
            unwrap_model(mpd).load_state_dict(discriminator_state["mpd"])
        if mrd is not None and discriminator_state.get("mrd") is not None:
            unwrap_model(mrd).load_state_dict(discriminator_state["mrd"])
        if optimizer_d is not None and discriminator_state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(discriminator_state["optimizer_d"])
        if scheduler_d is not None and discriminator_state.get("scheduler_d") is not None:
            scheduler_d.load_state_dict(discriminator_state["scheduler_d"])
        if scaler_d is not None and discriminator_state.get("scaler_d") is not None:
            scaler_d.load_state_dict(discriminator_state["scaler_d"])

    return (
        int(payload.get("epoch", 0)),
        int(payload.get("global_step", 0)),
        payload.get("best_val_loss"),
    )


__all__ = [
    "FrozenWav2Vec2Frontend",
    "apply_trainer_freeze_from_config",
    "build_bicodec_model",
    "cleanup_ddp",
    "ddp_average_metrics",
    "effective_trainer_freeze",
    "get_rank",
    "get_world_size",
    "initialize_training_model",
    "is_dist_initialized",
    "is_main_process",
    "make_optimizer",
    "maybe_load_resume",
    "resolve_training_paths",
    "save_checkpoint",
    "set_bicodec_experiment_model_train_mode",
    "set_seed",
    "setup_device_and_ddp",
    "speaker_xvector_dim",
    "unwrap_model",
]
