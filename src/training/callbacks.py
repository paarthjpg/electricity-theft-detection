"""Early stopping on validation PR-AUC (higher is better)."""
from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, epoch: int) -> bool:
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved
