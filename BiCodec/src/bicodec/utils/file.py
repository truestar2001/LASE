"""Compatibility wrappers for the legacy Spark-TTS-style utility imports."""

from bicodec.utils.config import load_config
from bicodec.utils.jsonl import read_jsonl, write_jsonl

__all__ = ["load_config", "read_jsonl", "write_jsonl"]
