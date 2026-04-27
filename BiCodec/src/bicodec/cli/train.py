import argparse
from pathlib import Path

from bicodec.training import train_bicodec
from bicodec.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or finetune BiCodec")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training config yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_bicodec(cfg)


if __name__ == "__main__":
    main()
