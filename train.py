import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import (
    RESULT_DIR,
    INIT_LR,
    WEIGHT_DECAY,
    EPOCHS,
    BACKBONE,
    SEED,
)
from data import load_datasets
from models import build_simple_cnn, build_efficientnet, build_silhouette_cnn, build_resnet, build_resnet34
from utils import plot_history, save_cls_report

os.makedirs(RESULT_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

train_ds, val_ds, test_ds, class_names = load_datasets(return_class_names=True)
num_classes = len(class_names)

if BACKBONE.lower() == "effnet":
    model = build_efficientnet(num_classes)
elif BACKBONE.lower() == "baseline":
    model = build_simple_cnn(num_classes)
elif BACKBONE.lower() == "resnet":
    model = build_resnet(num_classes)
elif BACKBONE.lower() == "resnet34":
    model = build_resnet34(num_classes)
else:
    model = build_silhouette_cnn(num_classes)

model.summary()

steps_per_epoch = sum(1 for _ in train_ds)
cosine = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=steps_per_epoch * EPOCHS,
    alpha=1e-2,
)
opt = AdamW(learning_rate=cosine, weight_decay=WEIGHT_DECAY)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
    ],
)

ckpt = ModelCheckpoint(
    os.path.join(RESULT_DIR, "best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
es = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
)
rlr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, es],
    # callbacks=[ckpt],
    verbose=1,
)

plot_history(history, prefix="train")

best_model_path = os.path.join(RESULT_DIR, "best.keras")
model = tf.keras.models.load_model(best_model_path)

print("\nEvaluating on test set using best weights …")
test_metrics = model.evaluate(test_ds, verbose=1)
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name}: {val:.4f}")

y_true, y_pred = [], []
for xb, yb in test_ds:
    logits = model.predict(xb, verbose=0)
    y_true.extend(yb.numpy().tolist())
    y_pred.extend(np.argmax(logits, axis=1).tolist())

# cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
# plot_confusion_matrix(cm, class_names, normalize=True, out="cm_norm.png")

save_cls_report(y_true, y_pred, class_names)

model.save(os.path.join(RESULT_DIR, f"final_model.keras"))
