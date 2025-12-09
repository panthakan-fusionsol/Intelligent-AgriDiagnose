import argparse
import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from corn_dataset import get_transforms, CornDiseaseDataset
from main import set_seed
import numpy as np
from sklearn.metrics import accuracy_score
import csv
import matplotlib_inline.backend_inline as mibi
mibi.set_matplotlib_formats("retina")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # ด้านบน

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_xticks_05(ax, rot=90, fs=7):
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))        # ระยะห่าง 0.05
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # แสดงเป็น 0.00
    ax.tick_params(axis='x', labelrotation=rot, labelsize=fs)



# ---------- helpers ----------
def find_intersection_x(x, y1, y2):
    x = np.asarray(x, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    d = y1 - y2
    s = np.sign(d)
    idx = np.where((s[:-1] == 0) | (s[:-1] * s[1:] < 0) | (s[1:] == 0))[0]
    if len(idx) > 0:
        i = int(idx[0])
        x0, x1 = x[i], x[i+1]
        d0, d1 = d[i], d[i+1]
        if d1 == d0:
            xi = x0
        else:
            t = -d0 / (d1 - d0)
            t = np.clip(t, 0.0, 1.0)
            xi = x0 + t * (x1 - x0)
        return float(xi), True
    j = int(np.argmin(np.abs(d)))
    return float(x[j]), False

def text2logs_time(file_path, text=""):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {text}"
    with open(p, "a", encoding="utf-8") as f:
        if not line.endswith("\n"):
            line += "\n"
        f.write(line)

def style_xticks(ax, rotation=90, fontsize=8):
    ax.tick_params(axis="x", labelrotation=rotation, labelsize=fontsize)

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["resnet18"], default="resnet18")
    parser.add_argument("--interval", type=float, default=0.025)
    parser.add_argument("--val_dir", default="/mnt/c/fusion/corn_project/All_Crops/validation")
    parser.add_argument("--classes", default="rust,blight,spot,virus,mildew,healthy")
    parser.add_argument("--state_path", default="/mnt/c/fusion/corn_project/model_training/resnet18_448_cosine_croponly_nojitter/checkpoints/best.pth")
    parser.add_argument("--img_size", type=int, default=448, choices=[224, 448])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="./eval_plots")
    parser.add_argument("--coverage", type=float, default=80.0)   # ให้เป็นตัวเลขชัดๆ
    parser.add_argument("--rmbg",default=False);
    args = parser.parse_args()
    set_seed(args.seed)

    # ----- Load checkpoint -----
    state = torch.load(args.state_path, map_location=device)
    class2id = state["class2id"]
    num_classes = state.get("num_classes", len(class2id))

    # ----- Build model -----
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()

    # ----- Prepare validation file list -----
    selected = set([s.strip().lower() for s in args.classes.split(",") if s.strip()])
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    val_paths, val_labels = [], []
    root = Path(args.val_dir)
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        cname = sub.name.split("_")[0].lower()
        if selected and (cname not in selected):
            continue
        if cname not in class2id:
            continue
        for p in sub.iterdir():
            if p.is_file() and p.suffix.lower() in img_exts:
                val_paths.append(p)
                val_labels.append(class2id[cname])

    assert len(val_paths) > 0, "No validation images found after filtering."

    _, tf = get_transforms(args.img_size, jitter="Off")
    ds = CornDiseaseDataset(val_paths, val_labels, tf,rmbg=args.rmbg)
    dl = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # ----- Sweep thresholds -----
    confs = np.round(np.arange(0.0, 1.0 + 1e-8, args.interval, dtype=np.float32), 6)
    accs, cover_pct = [], []

    with torch.inference_mode():
        for conf in tqdm(confs, desc="Threshold sweep"):
            all_preds, all_gts = [], []
            N, kept = 0, 0
            for X, y in dl:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(X)
                probs = torch.softmax(logits, dim=-1)
                max_prob, pred_idx = probs.max(dim=-1)
                idx = (max_prob >= conf)

                if idx.any():
                    all_preds.extend(pred_idx[idx].tolist())
                    all_gts.extend(y[idx].tolist())
                kept += int(idx.sum().item())
                N += X.size(0)

            if len(all_gts) == 0:
                acc = np.nan
                cov = 0.0
            else:
                acc = accuracy_score(all_gts, all_preds)
                cov = (kept / N) * 100.0 if N > 0 else 0.0

            accs.append(acc)
            cover_pct.append(cov)

    # ----- Save plots -----
    outdir = Path(args.output_path); outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "log.txt"

    confs_np = np.asarray(confs, dtype=np.float32)
    accs_np  = np.asarray(accs, dtype=np.float32)
    cov_pct  = np.asarray(cover_pct, dtype=np.float32)
    mask = ~np.isnan(accs_np)

    # ----- Save CSV file with confidence, accuracy, coverage -----
    csv_path = outdir / "confidence_accuracy_coverage.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['confidence', 'accuracy', 'coverage'])
        for conf_val, acc_val, cov_val in zip(confs_np, accs_np, cov_pct):
            # Handle NaN values by writing them as empty strings or 'NaN'
            acc_str = f"{acc_val:.6f}" if not np.isnan(acc_val) else "NaN"
            writer.writerow([f"{conf_val:.6f}", acc_str, f"{cov_val:.3f}"])
    print(f"CSV saved to: {csv_path}")

    # 1) Accuracy vs Confidence
    plt.figure(figsize=(7, 4.8))
    if mask.any():
        plt.plot(confs_np[mask], accs_np[mask], marker="o", linewidth=1, color="r", label="Accuracy")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    style_xticks(plt.gca(), rotation=90, fontsize=7)   # หมุน 90° + ฟอนต์เล็ก
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), framealpha=0.9)
    plt.tight_layout()
    fix_xticks_05(plt.gca())
    plt.savefig(outdir / "accuracy_vs_confidence.png" ,bbox_inches="tight")
    plt.close()

    # 2) % Images kept vs Confidence
    plt.figure(figsize=(7, 4.8))
    plt.plot(confs_np, cov_pct, marker="o", linewidth=1, color="b", label="% images kept")
    plt.xlabel("Confidence threshold")
    plt.ylabel("% images kept")
    plt.grid(True, alpha=0.3)
    style_xticks(plt.gca(), rotation=90, fontsize=7)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), framealpha=0.9)
    plt.tight_layout()
    fix_xticks_05(plt.gca())
    plt.savefig(outdir / "coverage_vs_confidence.png", bbox_inches="tight")
    plt.close()

    # ---------- Intersection (คำนวณเก็บค่า) ----------
    x_m   = confs_np[mask]
    acc_m = accs_np[mask]
    cov_f = (cov_pct / 100.0)
    cov_m = cov_f[mask]

    if len(x_m) >= 2:
        x_star, is_exact = find_intersection_x(x_m, acc_m, cov_m)
        acc_star = float(np.interp(x_star, x_m, acc_m))
        cov_star = float(np.interp(x_star, x_m, cov_m))
    else:
        x_star, is_exact = float(confs_np[mask][0]), False
        acc_star = float(acc_m[0]); cov_star = float(cov_m[0])

    # 3) Combined plot (สองเส้น)
    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    if mask.any():
        l1, = ax1.plot(x_m, acc_m, marker="o", linewidth=1, label="Accuracy", color='r', zorder=3)
    else:
        l1 = None
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy", color='r')
    ax1.tick_params(axis='y', colors='r')
    style_xticks(ax1, rotation=90, fontsize=7)
    ax1.grid(True, which="both", alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    l2, = ax2.plot(confs_np, cov_pct, marker="s", linewidth=1, label="% images kept", color='b', zorder=2)
    ax2.set_ylabel("% images kept", color='b')
    ax2.tick_params(axis='y', colors='b')

    lines = [l for l in (l1, l2) if l is not None]
    labels = [l.get_label() for l in lines]
    if lines:
        ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                   ncol=2, framealpha=0.9)
        
    fix_xticks_05(ax1)
    fig.savefig(outdir / "accuracy_and_coverage_vs_confidence.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- LOG ONLY: best_at_coverage>=target ----------
    target = float(args.coverage)
    eligible = np.where((cov_pct >= target) & mask)[0]
    if eligible.size > 0:
        deltas = cov_pct[eligible] - target
        min_delta = deltas.min()
        tie_idx = eligible[deltas == min_delta]
        best_idx = tie_idx[np.nanargmax(accs_np[tie_idx])]  # หากเสมอ เลือก acc สูงสุด
        conf_best = float(confs_np[best_idx])
        acc_best  = float(accs_np[best_idx])
        cov_best  = float(cov_pct[best_idx])
        text2logs_time(log_path,
            f"best_at_coverage>={int(target)}%: conf={conf_best:.6f} acc={acc_best:.6f} coverage%={cov_best:.3f}")
    else:
        text2logs_time(log_path,
            f"best_at_coverage>={int(target)}%: not found (all coverage below {int(target)}%)")

    # ---------- Log จุดตัด/ใกล้สุด ----------
    print(("intersection" if is_exact else "closest"),
          f"conf={x_star:.6f}",
          f"acc={acc_star:.6f}",
          f"coverage%={cov_star*100.0:.3f}")
    text2logs_time(log_path,
          f"{args.state_path}\nconf={x_star:.6f} acc={acc_star:.6f} coverage%={cov_star*100.0:.3f}")
