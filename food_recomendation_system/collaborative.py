from typing import Tuple
import numpy as np
import pandas as pd


def build_user_item_matrix(df: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'item_id', rating_col: str = 'rating') -> Tuple[np.ndarray, dict, dict]:
    users = df[user_col].unique().tolist()
    items = df[item_col].unique().tolist()
    u2i = {u: idx for idx, u in enumerate(users)}
    i2i = {i: idx for idx, i in enumerate(items)}
    mat = np.zeros((len(users), len(items)))
    for _, row in df.iterrows():
        u = u2i[row[user_col]]
        i = i2i[row[item_col]]
        mat[u, i] = row[rating_col]
    return mat, u2i, i2i


def cosine_sim_matrix(mat: np.ndarray):
    # rows are users or items
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    m = mat / norm
    return m @ m.T


def predict_user_based(R: np.ndarray, user_index: int, sim_matrix: np.ndarray, k: int = 5):
    sims = sim_matrix[user_index]
    # select top-k similar users
    idx = np.argsort(-sims)[1:k+1]
    num = sims[idx] @ R[idx]
    den = np.sum(np.abs(sims[idx])) + 1e-9
    return num / den


def predict_item_based(R: np.ndarray, item_index: int, sim_matrix: np.ndarray, k: int = 5):
    sims = sim_matrix[item_index]
    idx = np.argsort(-sims)[1:k+1]
    num = R[:, idx] @ sims[idx]
    den = np.sum(np.abs(sims[idx])) + 1e-9
    return num / den
