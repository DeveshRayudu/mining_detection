"""
CLI script to train and evaluate baseline models (Spectral Index Thresholding and Random Forest).
Evaluated on the Validation split to establish benchmark performance before deep learning.
Note: The frozen test set is preserved untouched for final unified evaluation.
"""

import json
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from src.config import config
from src.models import ThresholdBaseline, RandomForestBaseline
from src.utils.metrics import compute_binary_metrics


def run_baselines() -> None:
    print("=" * 70)
    print("🛰️  PHASE 4: BASELINE MODELS (THRESHOLDING & RANDOM FOREST)")
    print("=" * 70)

    # 1. Load Split Manifest
    manifest_path = config.dirs.data_dir / "split_manifest.json"
    if not manifest_path.exists():
        print(f"❌ Split manifest missing at {manifest_path}. Run Phase 3 first.")
        sys.exit(1)

    with open(manifest_path) as f:
        split_manifest = json.load(f)

    train_ids = split_manifest["train"]
    val_ids = split_manifest["val"]
    scene_id = "S2A_45QUE_20240102_0_L2A"

    print(f"Dataset Configuration:")
    print(f"  - Training tiles:   {len(train_ids)}")
    print(f"  - Validation tiles: {len(val_ids)}")
    print(f"  - Reference Scene:  {scene_id}")

    # Load arrays
    img_dir = config.dirs.data_tiles / scene_id
    mask_dir = config.dirs.data_labels / "tiles" / scene_id

    train_imgs = [np.load(img_dir / f"{tid}.npy") for tid in train_ids]
    train_masks = [np.load(mask_dir / f"{tid}.npy") for tid in train_ids]

    val_imgs = [np.load(img_dir / f"{tid}.npy") for tid in val_ids]
    val_masks = [np.load(mask_dir / f"{tid}.npy") for tid in val_ids]

    # ---------------------------------------------------------
    # Baseline 1: Spectral Index Thresholding
    # ---------------------------------------------------------
    print("\n[1/3] Optimizing & Evaluating Spectral Index Thresholding Baseline...")
    thresh_model = ThresholdBaseline()
    opt_results = thresh_model.optimize(val_imgs, val_masks)

    print(f"  Optimized Thresholds (on Val set):")
    print(f"    - NDVI Threshold: {thresh_model.ndvi_thresh:.3f} (Mine: NDVI < {thresh_model.ndvi_thresh:.3f})")
    print(f"    - BSI Threshold:  {thresh_model.bsi_thresh:.3f} (Mine: BSI  > {thresh_model.bsi_thresh:.3f})")
    print(f"    - MNDWI Max:      {thresh_model.mndwi_max:.3f} (Mine: MNDWI < {thresh_model.mndwi_max:.3f})")

    # Evaluate on Validation set
    thresh_preds = [thresh_model.predict(img) for img in val_imgs]
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    for pred, mask in zip(thresh_preds, val_masks):
        m = compute_binary_metrics(pred, mask)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_tn += m["tn"]

    thresh_metrics = {
        "iou": float(total_tp / (total_tp + total_fp + total_fn + 1e-7)),
        "f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7)),
        "precision": float(total_tp / (total_tp + total_fp + 1e-7)),
        "recall": float(total_tp / (total_tp + total_fn + 1e-7)),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }

    print(f"  📊 Validation Performance (Index Thresholding):")
    print(f"    - IoU (Jaccard): {thresh_metrics['iou']:.4f}")
    print(f"    - F1-Score:      {thresh_metrics['f1']:.4f}")
    print(f"    - Precision:     {thresh_metrics['precision']:.4f}")
    print(f"    - Recall:        {thresh_metrics['recall']:.4f}")

    # ---------------------------------------------------------
    # Baseline 2: Pixel-Level Random Forest
    # ---------------------------------------------------------
    print("\n[2/3] Training & Evaluating Pixel-Level Random Forest Baseline...")
    rf_model = RandomForestBaseline(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(train_imgs, train_masks, max_samples_per_class=50000)

    # Save RF model
    rf_save_path = config.dirs.models_dir / "baseline_rf.joblib"
    rf_model.save(rf_save_path)
    print(f"  Trained Random Forest (100 estimators) saved to {rf_save_path.name}")

    # Evaluate on Validation set
    rf_preds = [rf_model.predict(img, threshold=0.5) for img in val_imgs]
    rf_tp, rf_fp, rf_fn, rf_tn = 0, 0, 0, 0
    for pred, mask in zip(rf_preds, val_masks):
        m = compute_binary_metrics(pred, mask)
        rf_tp += m["tp"]
        rf_fp += m["fp"]
        rf_fn += m["fn"]
        rf_tn += m["tn"]

    rf_metrics = {
        "iou": float(rf_tp / (rf_tp + rf_fp + rf_fn + 1e-7)),
        "f1": float(2 * rf_tp / (2 * rf_tp + rf_fp + rf_fn + 1e-7)),
        "precision": float(rf_tp / (rf_tp + rf_fp + 1e-7)),
        "recall": float(rf_tp / (rf_tp + rf_fn + 1e-7)),
        "tp": rf_tp,
        "fp": rf_fp,
        "fn": rf_fn,
        "tn": rf_tn,
    }

    print(f"  📊 Validation Performance (Random Forest):")
    print(f"    - IoU (Jaccard): {rf_metrics['iou']:.4f}")
    print(f"    - F1-Score:      {rf_metrics['f1']:.4f}")
    print(f"    - Precision:     {rf_metrics['precision']:.4f}")
    print(f"    - Recall:        {rf_metrics['recall']:.4f}")

    # ---------------------------------------------------------
    # Save Benchmark Metrics
    # ---------------------------------------------------------
    print("\n[3/3] Saving Validation Baseline Summary...")
    baseline_summary = {
        "evaluation_split": "validation",
        "reference_scene": scene_id,
        "threshold_baseline": {
            "parameters": {
                "ndvi_thresh": thresh_model.ndvi_thresh,
                "bsi_thresh": thresh_model.bsi_thresh,
                "mndwi_max": thresh_model.mndwi_max,
            },
            "metrics": thresh_metrics,
        },
        "random_forest_baseline": {
            "parameters": {
                "n_estimators": 100,
                "max_depth": 12,
            },
            "metrics": rf_metrics,
        },
    }

    summary_path = config.dirs.outputs_dir / "baseline_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(baseline_summary, f, indent=2)

    print(f"  ✅ Saved benchmark comparisons to {summary_path.name}")
    print("\n" + "=" * 70)
    print("📊 BASELINE COMPARISON TABLE (Validation Set):")
    print(f"  {'Model':<25} | {'IoU':<8} | {'F1-Score':<8} | {'Precision':<10} | {'Recall':<8}")
    print(f"  {'-'*25}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}-|-{'-'*8}")
    print(f"  {'Index Thresholding':<25} | {thresh_metrics['iou']:<8.4f} | {thresh_metrics['f1']:<8.4f} | {thresh_metrics['precision']:<10.4f} | {thresh_metrics['recall']:<8.4f}")
    print(f"  {'Random Forest (100 trees)':<25} | {rf_metrics['iou']:<8.4f} | {rf_metrics['f1']:<8.4f} | {rf_metrics['precision']:<10.4f} | {rf_metrics['recall']:<8.4f}")
    print("=" * 70)
    print("🎉 PHASE 4 COMPLETE: Baselines Trained and Benchmarked on Validation Set!")
    print("=" * 70)


if __name__ == "__main__":
    run_baselines()
