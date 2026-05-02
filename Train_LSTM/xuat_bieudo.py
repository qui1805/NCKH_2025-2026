
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# =========================
# CONFIG
# =========================
OUT_DIR = r"C:\Train_LSTM\models"
FIG_DIR = os.path.join(OUT_DIR, "report_figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ["non_violence", "violence", "weapon"]
NUM_CLASSES = 3
DPI = 300


# =========================
# COMMON UTILS
# =========================
def save_current_fig(save_path: str):
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


def read_csv_if_exists(path: str):
    if not os.path.exists(path):
        print(f"[SKIP] Không tìm thấy file: {path}")
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json_if_exists(path: str):
    if not os.path.exists(path):
        print(f"[SKIP] Không tìm thấy file: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_value(metrics: dict, key: str, default=np.nan):
    if not isinstance(metrics, dict):
        return default
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


# =========================
# 1) TRAINING CURVES
# =========================
def plot_training_loss(history_df: pd.DataFrame):
    if "epoch" not in history_df.columns or "train_loss" not in history_df.columns:
        print("[SKIP] history thiếu cột epoch/train_loss")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    save_current_fig(os.path.join(FIG_DIR, "01_training_loss.png"))


def plot_validation_f1(history_df: pd.DataFrame):
    if "epoch" not in history_df.columns:
        print("[SKIP] history thiếu cột epoch")
        return

    plt.figure(figsize=(8, 5))

    has_curve = False
    if "val_f1_argmax" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["val_f1_argmax"], marker="o", linewidth=2, label="Argmax")
        has_curve = True

    if "val_f1_threshold" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["val_f1_threshold"], marker="s", linewidth=2, label="Threshold")
        has_curve = True

    if not has_curve:
        plt.close()
        print("[SKIP] history thiếu val_f1_argmax/val_f1_threshold")
        return

    plt.title("Validation Macro-F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_fig(os.path.join(FIG_DIR, "02_validation_macro_f1.png"))


# =========================
# 2) CONFUSION MATRIX
# =========================
def plot_confusion_matrix_from_csv(csv_path: str, save_name: str, title: str):
    if not os.path.exists(csv_path):
        print(f"[SKIP] Không tìm thấy confusion matrix: {csv_path}")
        return

    cm_df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    cm = cm_df.values.astype(int)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=35, ha="right")
    plt.yticks(range(len(cm_df.index)), cm_df.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    save_current_fig(os.path.join(FIG_DIR, save_name))


def plot_normalized_confusion_matrix_from_csv(csv_path: str, save_name: str, title: str):
    if not os.path.exists(csv_path):
        print(f"[SKIP] Không tìm thấy confusion matrix: {csv_path}")
        return

    cm_df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
    cm = cm_df.values.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=35, ha="right")
    plt.yticks(range(len(cm_df.index)), cm_df.index)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(label="Normalized value")
    save_current_fig(os.path.join(FIG_DIR, save_name))


# =========================
# 3) CLASS DISTRIBUTION
# =========================
def plot_class_distribution_from_final_metrics(final_metrics: dict):
    if not isinstance(final_metrics, dict):
        print("[SKIP] Không có final_metrics")
        return

    splits = ["train", "val", "test"]
    values = []
    ok = False

    for cls_name in CLASS_NAMES:
        row = []
        for split in splits:
            key = f"{split}_label_distribution"
            dist = final_metrics.get(key, {})
            row.append(int(dist.get(cls_name, 0)) if isinstance(dist, dict) else 0)
        values.append(row)
        if sum(row) > 0:
            ok = True

    if not ok:
        print("[SKIP] final_metrics không có *_label_distribution")
        return

    x = np.arange(len(splits))
    width = 0.25

    plt.figure(figsize=(8, 5))
    for idx, cls_name in enumerate(CLASS_NAMES):
        offset = (idx - 1) * width
        plt.bar(x + offset, values[idx], width=width, label=cls_name)

    plt.title("Class Distribution in Train/Validation/Test Sets")
    plt.xlabel("Dataset split")
    plt.ylabel("Number of samples")
    plt.xticks(x, ["Train", "Validation", "Test"])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    save_current_fig(os.path.join(FIG_DIR, "05_class_distribution.png"))


# =========================
# 4) TEST METRIC COMPARISON
# =========================
def plot_test_metric_comparison(final_metrics: dict):
    if not isinstance(final_metrics, dict):
        print("[SKIP] Không có final_metrics")
        return

    argmax_metrics = final_metrics.get("test_argmax", {})
    threshold_metrics = final_metrics.get("test_threshold", {})
    metric_names = ["acc", "precision", "recall", "f1"]

    argmax_values = [get_metric_value(argmax_metrics, m) for m in metric_names]
    threshold_values = [get_metric_value(threshold_metrics, m) for m in metric_names]

    if all(np.isnan(argmax_values)) and all(np.isnan(threshold_values)):
        print("[SKIP] final_metrics thiếu test_argmax/test_threshold")
        return

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, argmax_values, width=width, label="Argmax")
    plt.bar(x + width / 2, threshold_values, width=width, label="Threshold")
    plt.title("Test Metrics: Argmax vs Threshold")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "Macro-F1"])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    save_current_fig(os.path.join(FIG_DIR, "06_test_metrics_comparison.png"))


# =========================
# 5) PROBABILITY HISTOGRAM
# =========================
def plot_probability_histograms(pred_df: pd.DataFrame):
    prob_cols = ["p_non_violence", "p_violence", "p_weapon"]
    if not all(c in pred_df.columns for c in prob_cols):
        print("[SKIP] test_predictions thiếu cột xác suất")
        return

    for col, cls_name in zip(prob_cols, CLASS_NAMES):
        plt.figure(figsize=(8, 5))
        plt.hist(pred_df[col].values, bins=30, alpha=0.85)
        plt.title(f"Probability Distribution - {cls_name}")
        plt.xlabel(f"Predicted probability of {cls_name}")
        plt.ylabel("Number of samples")
        plt.grid(True, alpha=0.3)
        save_current_fig(os.path.join(FIG_DIR, f"07_prob_hist_{cls_name}.png"))


# =========================
# 6) ROC CURVE ONE-VS-REST
# =========================
def plot_roc_curves(pred_df: pd.DataFrame):
    required = ["y_true", "p_non_violence", "p_violence", "p_weapon"]
    if not all(c in pred_df.columns for c in required):
        print("[SKIP] Không đủ cột để vẽ ROC")
        return

    y_true = pred_df["y_true"].astype(int).values
    prob_cols = ["p_non_violence", "p_violence", "p_weapon"]

    plt.figure(figsize=(8, 6))
    plotted = False

    for class_id, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_true == class_id).astype(int)
        scores = pred_df[prob_cols[class_id]].values

        if len(np.unique(y_bin)) < 2:
            print(f"[SKIP] ROC {cls_name}: y_true không có đủ positive/negative")
            continue

        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{cls_name} (AUC={roc_auc:.3f})")
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.title("ROC Curve - One-vs-Rest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_fig(os.path.join(FIG_DIR, "08_roc_curve_ovr.png"))


# =========================
# 7) PRECISION-RECALL CURVE
# =========================
def plot_pr_curves(pred_df: pd.DataFrame):
    required = ["y_true", "p_non_violence", "p_violence", "p_weapon"]
    if not all(c in pred_df.columns for c in required):
        print("[SKIP] Không đủ cột để vẽ Precision-Recall")
        return

    y_true = pred_df["y_true"].astype(int).values
    prob_cols = ["p_non_violence", "p_violence", "p_weapon"]

    plt.figure(figsize=(8, 6))
    plotted = False

    for class_id, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_true == class_id).astype(int)
        scores = pred_df[prob_cols[class_id]].values

        if len(np.unique(y_bin)) < 2:
            print(f"[SKIP] PR {cls_name}: y_true không có đủ positive/negative")
            continue

        precision, recall, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        plt.plot(recall, precision, linewidth=2, label=f"{cls_name} (AP={ap:.3f})")
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title("Precision-Recall Curve - One-vs-Rest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_fig(os.path.join(FIG_DIR, "09_precision_recall_curve_ovr.png"))


# =========================
# 8) ERROR ANALYSIS BAR
# =========================
def plot_error_summary(pred_df: pd.DataFrame):
    required = ["y_true", "pred_argmax", "pred_threshold"]
    if not all(c in pred_df.columns for c in required):
        print("[SKIP] Không đủ cột để vẽ error summary")
        return

    y_true = pred_df["y_true"].astype(int).values
    pred_argmax = pred_df["pred_argmax"].astype(int).values
    pred_threshold = pred_df["pred_threshold"].astype(int).values

    argmax_wrong = int((y_true != pred_argmax).sum())
    threshold_wrong = int((y_true != pred_threshold).sum())

    plt.figure(figsize=(7, 5))
    plt.bar(["Argmax", "Threshold"], [argmax_wrong, threshold_wrong])
    plt.title("Number of Misclassified Test Samples")
    plt.xlabel("Decision method")
    plt.ylabel("Number of wrong predictions")
    plt.grid(True, axis="y", alpha=0.3)
    save_current_fig(os.path.join(FIG_DIR, "10_error_summary.png"))


# =========================
# MAIN
# =========================
def main():
    warnings.filterwarnings("ignore")

    history_path = os.path.join(OUT_DIR, "history_threshold_train.csv")
    pred_path = os.path.join(OUT_DIR, "test_predictions_with_probs.csv")
    final_metrics_path = os.path.join(OUT_DIR, "final_metrics_threshold_train.json")
    cm_argmax_path = os.path.join(OUT_DIR, "conf_matrix_test_argmax.csv")
    cm_threshold_path = os.path.join(OUT_DIR, "conf_matrix_test_threshold.csv")

    history_df = read_csv_if_exists(history_path)
    pred_df = read_csv_if_exists(pred_path)
    final_metrics = read_json_if_exists(final_metrics_path)

    if history_df is not None:
        plot_training_loss(history_df)
        plot_validation_f1(history_df)

    plot_confusion_matrix_from_csv(
        cm_argmax_path,
        "03_confusion_matrix_argmax.png",
        "Confusion Matrix - Test Set (Argmax)",
    )
    plot_confusion_matrix_from_csv(
        cm_threshold_path,
        "04_confusion_matrix_threshold.png",
        "Confusion Matrix - Test Set (Threshold)",
    )
    plot_normalized_confusion_matrix_from_csv(
        cm_threshold_path,
        "04b_confusion_matrix_threshold_normalized.png",
        "Normalized Confusion Matrix - Test Set (Threshold)",
    )

    if final_metrics is not None:
        plot_class_distribution_from_final_metrics(final_metrics)
        plot_test_metric_comparison(final_metrics)

    if pred_df is not None:
        plot_probability_histograms(pred_df)
        plot_roc_curves(pred_df)
        plot_pr_curves(pred_df)
        plot_error_summary(pred_df)

    print("\n=== DONE EXPORT REPORT FIGURES ===")
    print(f"[INFO] Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
