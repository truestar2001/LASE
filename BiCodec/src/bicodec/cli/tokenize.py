import argparse
from pathlib import Path

import torch

from bicodec.models import BiCodecTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize an audio file with BiCodec")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Artifact root")
    parser.add_argument("--audio-path", type=Path, required=True, help="Input wav path")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file path")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda:0 or cpu. Defaults to auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    tokenizer = BiCodecTokenizer(args.artifacts_dir, device=device)
    global_tokens, semantic_tokens = tokenizer.tokenize(str(args.audio_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_tokens": global_tokens.cpu(),
            "semantic_tokens": semantic_tokens.cpu(),
            "audio_path": str(args.audio_path),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
