import os
import argparse

parser = argparse.ArgumentParser()

# Data
IMG_SIZE = (128, 128)
CHANNELS = 1
BATCH = 64
TRAIN_SIZE = 8  # number of training data per test image (before train/validation split)
VAL_SPLIT = 0.0  # in range of 0-1

EPOCHS = 60
INIT_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
parser.add_argument(
    "--BACKBONE", type=str, default="baseline", help="Model backbone to use."
)  # "baseline" or "effnet" or "resnet"
parser.add_argument(
    "--MODEL", type=str, default=None, help="EfficientNet backbone to use."
)  # "b0", "b1", ..., "b7" for EffNet; 18/34/50 for ResNet
args = parser.parse_args()
BACKBONE = args.BACKBONE.lower()
MODEL = args.MODEL.lower() if args.MODEL else None


TRAINABLE_AT = int(
    os.getenv("TRAINABLE_AT", "150")
)  # fine-tuning depth (0 = freeze all, higher = unfreeze more)
SEED = 42

TEST_DIR = os.getenv("TEST_DIR", "data/testset")
RESULT_DIR = (
    "result/{BACKBONE}" if BACKBONE == "baseline" else f"result/{BACKBONE}/{MODEL}"
)
