import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from config import RESULT_DIR

os.makedirs(RESULT_DIR, exist_ok=True)


def plot_history(h, prefix="train"):
    plt.figure()
    plt.plot(h.history["accuracy"], label="acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.savefig(os.path.join(RESULT_DIR, f"{prefix}_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(h.history["loss"], label="loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.savefig(os.path.join(RESULT_DIR, f"{prefix}_loss.png"))
    plt.close()


def plot_confusion_matrix(
    cm, classes, normalize=True, out="cm_norm.png", title="Confusion matrix"
):
    cm2 = cm.astype("float")
    if normalize:
        cm2 /= cm2.sum(axis=1, keepdims=True) + 1e-9

    plt.figure(figsize=(max(6, 0.4 * len(classes)), max(5, 0.4 * len(classes))))
    plt.imshow(cm2, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f"
    thresh = cm2.max() * 0.6
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(
            j,
            i,
            format(cm2[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm2[i, j] > thresh else "black",
            fontsize=7,
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{out}"))
    plt.close()


def save_cls_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(RESULT_DIR, f"class_report.txt"), "w") as f:
        f.write(rep)
    print(rep)
