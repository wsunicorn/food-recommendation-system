import numpy as np


def weighted_hybrid(pred_cb: np.ndarray, pred_cf: np.ndarray, alpha: float = 0.5):
    return alpha * pred_cb + (1 - alpha) * pred_cf
