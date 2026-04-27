import argparse
from pathlib import Path

import soundfile as sf
import torch

from bicodec.models import BiCodecTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct waveform from BiCodec tokens")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Artifact root")
    parser.add_argument("--tokens-path", type=Path, required=True, help="Input .pt token file")
    parser.add_argument("--output-wav", type=Path, required=True, help="Output wav path")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda:0 or cpu. Defaults to auto.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.tokens_path, map_location="cpu", weights_only=False)
    global_tokens = payload["global_tokens"]
    semantic_tokens = payload["semantic_tokens"]

    if global_tokens.dim() == 2 and global_tokens.shape[0] == 1:
        global_tokens = global_tokens.squeeze(0)

    device = torch.device(args.device) if args.device else None
    tokenizer = BiCodecTokenizer(args.artifacts_dir, device=device)
    wav = tokenizer.detokenize(global_tokens.to(tokenizer.device), semantic_tokens.to(tokenizer.device))
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, wav, args.sample_rate)


if __name__ == "__main__":
    main()
