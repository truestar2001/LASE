from importlib import import_module

__all__ = ["load_config", "read_jsonl", "resolve_artifact_layout", "write_jsonl"]


def __getattr__(name: str):
    if name == "load_config":
        return import_module("bicodec.utils.config").load_config
    if name == "read_jsonl":
        return import_module("bicodec.utils.jsonl").read_jsonl
    if name == "write_jsonl":
        return import_module("bicodec.utils.jsonl").write_jsonl
    if name == "resolve_artifact_layout":
        return import_module("bicodec.utils.layout").resolve_artifact_layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
