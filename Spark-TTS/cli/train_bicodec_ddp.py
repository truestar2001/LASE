import argparse

from sparktts.training.bicodec_trainer import train_bicodec
from sparktts.utils.file import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train or finetune Spark-TTS BiCodec")
    parser.add_argument(
        "--config",
        default="/home/jyp/LASE/Spark-TTS/configs/train_bicodec.yaml",
        help="Path to a training config yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_bicodec(cfg)


if __name__ == "__main__":
    main()
