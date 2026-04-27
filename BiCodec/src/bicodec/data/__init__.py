from importlib import import_module

__all__ = [
    "AudioManifestDataset",
    "build_reference_clip",
    "build_speaker_label_map",
    "collate_audio_batch",
    "load_manifest",
    "pad_1d_batch",
    "split_manifest_by_speaker",
]


def __getattr__(name: str):
    if name in {"collate_audio_batch", "pad_1d_batch"}:
        module = import_module("bicodec.data.collate")
        return getattr(module, name)
    if name in {"AudioManifestDataset", "build_reference_clip"}:
        module = import_module("bicodec.data.dataset")
        return getattr(module, name)
    if name in {"build_speaker_label_map", "load_manifest"}:
        module = import_module("bicodec.data.manifest")
        return getattr(module, name)
    if name == "split_manifest_by_speaker":
        module = import_module("bicodec.data.split")
        return module.split_manifest_by_speaker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
