import os

# Data
IMG_SIZE = (128, 128)
CHANNELS = 1
BATCH = 64
TRAIN_SIZE = 8  # number of training data per test image (before train/validation split)
VAL_SPLIT = 0.0  # in range of 0-1

# Training
EPOCHS = 60
INIT_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
BACKBONE = os.getenv("BACKBONE", "baseline")  # or "silhouette" or "effnet"

# EfficientNet fine-tuning depth (0 = freeze all, higher = unfreeze more)
TRAINABLE_AT = int(os.getenv("TRAINABLE_AT", "175"))
SEED = 42

# Paths
AUG_PKL_PATH = os.getenv("AUG_PKL_PATH", "data/X_aug_y_aug.pkl")
TEST_DIR = os.getenv("TEST_DIR", "data/testset")
RESULT_DIR = os.getenv("RESULT_DIR", f"result/{BACKBONE}")
