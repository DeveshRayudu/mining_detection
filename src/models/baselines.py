"""
Baseline Mining Detection Models:
1. Empirical Spectral Index Thresholding (NDVI, BSI, MNDWI)
2. Pixel-Level Random Forest Classifier
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.metrics import compute_binary_metrics


class ThresholdBaseline:
    """Rule-based mining detection using remote sensing spectral indices."""

    def __init__(
        self,
        ndvi_thresh: float = 0.25,
        bsi_thresh: float = 0.05,
        mndwi_max: float = 0.0,
    ):
        self.ndvi_thresh = ndvi_thresh
        self.bsi_thresh = bsi_thresh
        self.mndwi_max = mndwi_max

    def predict(self, feature_stack: np.ndarray) -> np.ndarray:
        """
        Predict binary mine mask from 8-channel input stack [B02..B12, NDVI, BSI].
        
        Args:
            feature_stack: Array of shape (8, H, W)
            
        Returns:
            Binary mask (H, W) where 1 indicates mining, 0 otherwise.
        """
        assert feature_stack.shape[0] == 8, f"Expected 8 channels, got {feature_stack.shape[0]}"
        
        green = feature_stack[1]
        swir1 = feature_stack[4]
        ndvi = feature_stack[6]
        bsi = feature_stack[7]

        # Calculate MNDWI: (Green - SWIR1) / (Green + SWIR1 + eps)
        mndwi_denom = green + swir1
        mndwi = np.where(mndwi_denom > 1e-8, (green - swir1) / (mndwi_denom + 1e-8), 0.0)

        # Mining decision rule: low vegetation, high bare ground, non-water
        mine_mask = (ndvi < self.ndvi_thresh) & (bsi > self.bsi_thresh) & (mndwi < self.mndwi_max)
        return mine_mask.astype(np.uint8)

    def optimize(
        self,
        val_images: List[np.ndarray],
        val_masks: List[np.ndarray],
    ) -> Dict[str, float]:
        """Grid searches optimal NDVI and BSI thresholds on the validation set."""
        best_f1 = -1.0
        best_params = {"ndvi_thresh": self.ndvi_thresh, "bsi_thresh": self.bsi_thresh}

        # Grid search over physically plausible ranges
        for ndvi_t in np.linspace(0.15, 0.40, 6):
            for bsi_t in np.linspace(-0.05, 0.15, 5):
                self.ndvi_thresh = float(ndvi_t)
                self.bsi_thresh = float(bsi_t)

                total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
                for img, mask in zip(val_images, val_masks):
                    pred = self.predict(img)
                    m = compute_binary_metrics(pred, mask)
                    total_tp += m["tp"]
                    total_fp += m["fp"]
                    total_fn += m["fn"]
                    total_tn += m["tn"]

                f1 = float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7))
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"ndvi_thresh": float(ndvi_t), "bsi_thresh": float(bsi_t)}

        self.ndvi_thresh = best_params["ndvi_thresh"]
        self.bsi_thresh = best_params["bsi_thresh"]
        return {"best_f1": best_f1, **best_params}


class RandomForestBaseline:
    """Pixel-level Random Forest Classifier using 8 spectral/index features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(
        self,
        train_images: List[np.ndarray],
        train_masks: List[np.ndarray],
        max_samples_per_class: int = 50000,
    ) -> None:
        """
        Samples balanced pixels across training patches and fits the Random Forest.
        """
        assert len(train_images) == len(train_masks)
        pos_features: List[np.ndarray] = []
        neg_features: List[np.ndarray] = []

        for img, mask in zip(train_images, train_masks):
            # Reshape features to (N_pixels, 8)
            feat_pixels = np.transpose(img, (1, 2, 0)).reshape(-1, 8)
            mask_pixels = mask.ravel()

            pos_idx = np.where(mask_pixels == 1)[0]
            neg_idx = np.where(mask_pixels == 0)[0]

            if len(pos_idx) > 0:
                pos_features.append(feat_pixels[pos_idx])
            if len(neg_idx) > 0:
                neg_features.append(feat_pixels[neg_idx])

        all_pos = np.concatenate(pos_features, axis=0) if pos_features else np.empty((0, 8))
        all_neg = np.concatenate(neg_features, axis=0) if neg_features else np.empty((0, 8))

        # Balanced subsampling
        n_pos = min(len(all_pos), max_samples_per_class)
        n_neg = min(len(all_neg), max_samples_per_class)

        rng = np.random.RandomState(42)
        pos_sub = all_pos[rng.choice(len(all_pos), n_pos, replace=False)] if len(all_pos) > 0 else all_pos
        neg_sub = all_neg[rng.choice(len(all_neg), n_neg, replace=False)] if len(all_neg) > 0 else all_neg

        x_train = np.vstack([pos_sub, neg_sub])
        y_train = np.hstack([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])

        self.rf.fit(x_train, y_train)
        self.is_fitted = True

    def predict(self, feature_stack: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Infers mine probability map and returns binary mask (H, W).
        """
        assert self.is_fitted, "Model must be fitted before predict"
        h, w = feature_stack.shape[1], feature_stack.shape[2]
        feat_pixels = np.transpose(feature_stack, (1, 2, 0)).reshape(-1, 8)

        probs = self.rf.predict_proba(feat_pixels)[:, 1]
        prob_map = probs.reshape(h, w)
        return (prob_map >= threshold).astype(np.uint8)

    def save(self, path: Path) -> None:
        """Serializes trained model to joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf, path)

    def load(self, path: Path) -> None:
        """Loads trained model from joblib."""
        self.rf = joblib.load(path)
        self.is_fitted = True
