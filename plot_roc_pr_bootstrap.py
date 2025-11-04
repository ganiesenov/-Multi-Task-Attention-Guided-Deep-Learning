script_path = '/mnt/data/plot_roc_pr_bootstrap.py'
script_content = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-fold ROC/PR with bootstrap CIs.
Input CSV format (header required):
    fold,id,y_true,y_pred
Where:
    - fold: integer or string fold id (e.g., 1..5)
    - id: sample identifier (optional)
    - y_true: 0/1 ground truth (binary)
    - y_pred: predicted probability for the positive class (float in [0,1])
Usage:
    python plot_roc_pr_bootstrap.py --csv preds.csv --outdir out --n_boot 2000 --spec 0.90
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
rng = np.random.default_rng(12345)

def interp_curve(x, y, grid):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    eps = 1e-12
    x_sorted = np.maximum.accumulate(x_sorted + np.linspace(0, eps, len(x_sorted)))
    return np.interp(grid, x_sorted, y_sorted)

def bootstrap_ci(arr, alpha=0.05):
    lo = np.percentile(arr, 100*(alpha/2), axis=0)
    hi = np.percentile(arr, 100*(1-alpha/2), axis=0)
    return lo, hi

def compute_operating_point(y_true, y_pred, target_spec=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    spec = 1.0 - fpr
    idx = int(np.argmin(np.abs(spec - target_spec)))
    return fpr[idx], tpr[idx], thr[idx], spec[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--spec", type=float, default=0.90)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    for col in ["fold","y_true","y_pred"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in CSV.")
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(float)

    # Per-fold curves and metrics
    fpr_grid = np.linspace(0,1,1001)
    rec_grid = np.linspace(0,1,1001)

    aucs, aps = [], []
    perfold_tpr, perfold_prec = [], []
    fold_order = []
    for fold, g in df.groupby("fold"):
        fold_order.append(fold)
        y = g["y_true"].values
        p = g["y_pred"].values
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        aucs.append(auc(fpr, tpr))
        aps.append(average_precision_score(y, p))
        perfold_tpr.append(interp_curve(fpr, tpr, fpr_grid))
        perfold_prec.append(interp_curve(rec, prec, rec_grid))

    mean_tpr = np.mean(np.stack(perfold_tpr, axis=0), axis=0)
    mean_prec = np.mean(np.stack(perfold_prec, axis=0), axis=0)

    # Bootstrap pooled CI
    y_all = df["y_true"].values
    p_all = df["y_pred"].values
    n = len(y_all)
    boot_tpr, boot_prec, boot_auc, boot_ap = [], [], [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_all[idx], p_all[idx]
        fpr_b, tpr_b, _ = roc_curve(yb, pb)
        prec_b, rec_b, _ = precision_recall_curve(yb, pb)
        boot_tpr.append(interp_curve(fpr_b, tpr_b, fpr_grid))
        boot_prec.append(interp_curve(rec_b, prec_b, rec_grid))
        boot_auc.append(auc(fpr_b, tpr_b))
        boot_ap.append(average_precision_score(yb, pb))
    boot_tpr = np.stack(boot_tpr, axis=0)
    boot_prec = np.stack(boot_prec, axis=0)
    tpr_lo, tpr_hi = bootstrap_ci(boot_tpr, alpha=args.alpha)
    prec_lo, prec_hi = bootstrap_ci(boot_prec, alpha=args.alpha)
    auc_lo, auc_hi = np.percentile(boot_auc, [100*(args.alpha/2), 100*(1-args.alpha/2)])
    ap_lo, ap_hi = np.percentile(boot_ap, [100*(args.alpha/2), 100*(1-args.alpha/2)])

    # Operating point
    op_fpr, op_tpr, op_thr, op_spec = compute_operating_point(y_all, p_all, target_spec=args.spec)

    # Plot ROC
    plt.figure(figsize=(5,4), dpi=200)
    for fold, tpr_int in zip(fold_order, perfold_tpr):
        plt.plot(fpr_grid, tpr_int, linewidth=0.8, alpha=0.8, label=f"Fold {fold}")
    plt.plot(fpr_grid, mean_tpr, linewidth=2.0, label="Mean ROC")
    plt.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, label="95% CI")
    plt.scatter([op_fpr], [op_tpr], s=20, label=f"@ {op_spec:.0%} spec")
    plt.plot([0,1],[0,1], linestyle="--", linewidth=0.8)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Per-fold ROC with 95% CI")
    plt.legend(loc="lower right", fontsize=7)
    roc_path = os.path.join(args.outdir, "roc_perfold_ci.png")
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches="tight")

    # Plot PR
    plt.figure(figsize=(5,4), dpi=200)
    for fold, prec_int in zip(fold_order, perfold_prec):
        plt.plot(rec_grid, prec_int, linewidth=0.8, alpha=0.8, label=f"Fold {fold}")
    plt.plot(rec_grid, mean_prec, linewidth=2.0, label="Mean PR")
    plt.fill_between(rec_grid, prec_lo, prec_hi, alpha=0.2, label="95% CI")
    baseline = y_all.sum() / len(y_all) if len(y_all) else 0.0
    plt.hlines(baseline, 0, 1, linestyles="--", linewidth=0.8, label=f"Baseline={baseline:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-fold Precision–Recall with 95% CI")
    plt.legend(loc="lower left", fontsize=7)
    pr_path = os.path.join(args.outdir, "pr_perfold_ci.png")
    plt.tight_layout()
    plt.savefig(pr_path, bbox_inches="tight")

    # Metrics summary
    lines = []
    lines.append("=== Per-fold metrics ===")
    for fold, a, ap in zip(fold_order, aucs, aps):
        lines.append(f"Fold {fold}: AUC={a:.4f}, AP={ap:.4f}")
    lines.append("")
    lines.append(f"Mean±SD AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    lines.append(f"Mean±SD AP : {np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}")
    lines.append("")
    lines.append(f"Bootstrap pooled AUC 95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]")
    lines.append(f"Bootstrap pooled AP  95% CI: [{ap_lo:.4f}, {ap_hi:.4f}]")
    lines.append("")
    lines.append(f"Operating point @ {args.spec:.0%} specificity: TPR={op_tpr:.3f}, FPR={op_fpr:.3f}, thr={op_thr:.4f}")
    out_path = os.path.join(args.outdir, "metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {roc_path}\nSaved: {pr_path}\nSaved: {out_path}")

if __name__ == "__main__":
    main()
'''
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script_content)
script_path
