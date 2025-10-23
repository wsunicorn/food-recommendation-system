import numpy as np
from typing import Tuple


class MatrixFactorization:
    def __init__(self, n_users: int, n_items: int, k: int = 20, lr: float = 0.01, reg: float = 0.01, epochs: int = 20):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.P = np.random.normal(scale=0.1, size=(n_users, k))
        self.Q = np.random.normal(scale=0.1, size=(n_items, k))

    def train(self, interactions: list):
        # interactions: list of (u, i, r)
        for epoch in range(self.epochs):
            for u, i, r in interactions:
                pred = self.P[u].dot(self.Q[i])
                err = r - pred
                # gradient update
                p_u = self.P[u]
                q_i = self.Q[i]
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)

    def predict(self, u: int, i: int) -> float:
        return self.P[u].dot(self.Q[i])

    def predict_user(self, u: int):
        return self.Q.dot(self.P[u])
