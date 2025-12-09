# utils.py
from __future__ import annotations
import os
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
import seaborn as sns


# -----------------------------
# Path / time helpers
# -----------------------------
def ensure_dir(d: Path | str) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d

def now_stamp() -> str:
    # ex: 1509251420
    return datetime.now().strftime("%d%m%y%H%M%S")


# -----------------------------
# Image helpers
# -----------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm_image(img: torch.Tensor,
                 mean: torch.Tensor = _IMAGENET_MEAN,
                 std: torch.Tensor = _IMAGENET_STD) -> np.ndarray:
    """
    img: torch.Tensor(C,H,W) on CPU or GPU, [normalized]
    return: np.ndarray(H,W,C) in [0,1]
    """
    if img.is_cuda:
        img = img.detach().cpu()
    x = img * std + mean
    x = torch.clamp(x, 0, 1)
    return x.numpy().transpose(1, 2, 0)


# -----------------------------
# Evaluation helpers
# -----------------------------
@torch.no_grad()
def evaluate_on_loader(model: torch.nn.Module,
                       loader,
                       device: torch.device) -> Tuple[List[int], List[int]]:
    """
    Return y_true, y_pred over the given loader
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return y_true, y_pred


# -----------------------------
# Plots: confusion matrix & curves
# -----------------------------
def plot_confusion_matrix(y_true: List[int],
                          y_pred: List[int],
                          class_names: List[str],
                          normalize: bool = False,   # <-- ค่าเริ่มต้นเป็น False = นับจริง (จำนวนเต็ม)
                          save_path: Optional[Path | str] = None,
                          writer=None,
                          tag: str = "Figures/ConfusionMatrix",
                          step: Optional[int] = None):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(
        y_true, y_pred,
        labels=labels,
        normalize=("true" if normalize else None)
    )

    # ถ้าไม่ normalize ให้เป็นจำนวนเต็มแน่นอน
    if not normalize:
        cm = cm.astype(int)
        fmt = "d"
        title_suffix = ""
    else:
        fmt = ".2f"
        title_suffix = " (normalized)"

    plt.figure(figsize=(1.0 + 0.6 * len(class_names), 1.0 + 0.6 * len(class_names)))
    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix{title_suffix}")
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if writer is not None:
        writer.add_figure(tag, plt.gcf(), global_step=0 if step is None else step)
    plt.close()



def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         train_accs: List[float],
                         val_accs: List[float],
                         save_dir: Path | str,
                         timestamp: Optional[str] = None,
                         writer=None):
    save_dir = ensure_dir(save_dir)
    ts = timestamp or now_stamp()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = save_dir / f"loss_curve_{ts}.png"
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    if writer is not None:
        writer.add_figure("Figures/LossCurve", plt.gcf(), global_step=len(train_losses))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = save_dir / f"accuracy_curve_{ts}.png"
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    if writer is not None:
        writer.add_figure("Figures/AccuracyCurve", plt.gcf(), global_step=len(train_accs))
    plt.close()

    return str(loss_path), str(acc_path)


# -----------------------------
# Visualization of predictions
# -----------------------------
@torch.no_grad()
def visualize_predictions(model: torch.nn.Module,
                          loader,
                          class_names: List[str],
                          device: torch.device,
                          num_samples: int = 8,
                          save_path: Optional[Path | str] = None,
                          writer=None,
                          tag: str = "Figures/SamplePredictions",
                          step: Optional[int] = None) -> List[Dict]:
    """
    Show first `num_samples` predictions from loader; save to file; return per-image dicts.
    """
    model.eval()
    samples = 0
    results: List[Dict] = []

    cols = min(5, max(1, num_samples))
    rows = math.ceil(num_samples / cols)
    fig_w, fig_h = cols * 3.2, rows * 3.5
    plt.figure(figsize=(fig_w, fig_h))

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        bs = data.size(0)
        for i in range(bs):
            if samples >= num_samples:
                break

            img_np = denorm_image(data[i])
            actual = class_names[targets[i].item()]
            pred = class_names[preds[i].item()]
            conf = probs[i, preds[i].item()].item()
            correct = (preds[i].item() == targets[i].item())

            results.append({
                "image_index": samples + 1,
                "actual": actual,
                "predicted": pred,
                "confidence": conf,
                "correct": correct
            })

            plt.subplot(rows, cols, samples + 1)
            plt.imshow(img_np)
            color = "green" if correct else "red"
            plt.title(f"\nActual: {actual}\nPred: {pred}\nConf: {conf:.3f}", color=color, fontsize=8)
            plt.axis("off")
            samples += 1

        if samples >= num_samples:
            break

    plt.suptitle(f"Sample Predictions ({num_samples})  -  Green=Correct, Red=Incorrect\n", fontsize=12)
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if writer is not None:
        writer.add_figure(tag, plt.gcf(), global_step=0 if step is None else step)
    plt.close()
    return results


@torch.no_grad()
def visualize_misclassifications(model: torch.nn.Module,
                                 loader,
                                 class_names: List[str],
                                 device: torch.device,
                                 max_samples: int = 20,
                                 save_path: Optional[Path | str] = None,
                                 writer=None,
                                 tag: str = "Figures/Misclassifications",
                                 step: Optional[int] = None) -> List[Dict]:
    """
    Collect up to `max_samples` misclassified items, make a grid image, save & return list.
    """
    model.eval()
    mis: List[Dict] = []

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(data.size(0)):
            if preds[i].item() != targets[i].item():
                img_np = denorm_image(data[i])
                actual = class_names[targets[i].item()]
                pred = class_names[preds[i].item()]
                conf = probs[i, preds[i].item()].item()

                mis.append({
                    "image": img_np,
                    "actual": actual,
                    "predicted": pred,
                    "confidence": conf
                })
                if len(mis) >= max_samples:
                    break
        if len(mis) >= max_samples:
            break

    if len(mis) == 0:
        # still write an empty artifact for traceability if save_path provided
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.figure(figsize=(6, 3))
            plt.axis("off")
            plt.title("No misclassified samples found.")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return []

    cols = min(5, max(1, len(mis)))
    rows = math.ceil(len(mis) / cols)
    plt.figure(figsize=(cols * 3.2, rows * 3.5))
    for idx, item in enumerate(mis):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(item["image"])
        plt.title(f"\nActual: {item['actual']}\nPred: {item['predicted']}\nConf: {item['confidence']:.3f}",
                  color="red", fontsize=8)
        plt.axis("off")
    plt.suptitle(f"Misclassified Samples ({len(mis)})\n", fontsize=12)
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if writer is not None:
        writer.add_figure(tag, plt.gcf(), global_step=0 if step is None else step)
    plt.close()
    return mis


# -----------------------------
# Save / Inference helpers
# -----------------------------
def save_model(model: torch.nn.Module,
               filepath: Path | str,
               class_names: List[str],
               class_to_idx: Dict[str, int],
               timestamp: str,
               training_info: Dict,
               architechture):
    state = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "architecture": architechture,
        "timestamp": timestamp,
        "training_info": training_info,
    }
    ensure_dir(Path(filepath).parent)
    torch.save(state, filepath)

@torch.no_grad()
def predict_single_image(model: torch.nn.Module,
                         image_path: Path | str,
                         transform,
                         class_names: List[str],
                         device: torch.device) -> Tuple[str, float, np.ndarray]:
    """
    Predict class for a single image.
    Return: (predicted_class, confidence, probs_array)
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)
    pred_name = class_names[pred_idx.item()]
    return pred_name, conf.item(), probs[0].detach().cpu().numpy()
