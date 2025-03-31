__all__ = ["PrintingCallback", "TqdmCallback"]

from typing import Dict, List

from ray.train import Callback
from tqdm import tqdm


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 10,
        minimize: bool = True,
        verbose: bool = False,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait_count = 0
        self.minimize = minimize
        self.verbose = verbose
        self.curr_best = float("inf") if minimize else float("-inf")

        super().__init__()

    def handle_result(self, results: Dict, **info):
        """Called when a trial reports a result."""
        avg_result = results[self.monitor]

        delta = avg_result - self.curr_best
        delta = -1 * delta if self.minimize else delta

        if delta > self.min_delta:
            self.curr_best = avg_result
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count > self.patience:
            print("STOP") if self.verbose else None


class PrintingCallback(Callback):
    def handle_result(self, results: Dict, **info):
        """Called when a trial reports a result."""
        print(results)


class TqdmCallback(Callback):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
        self.epoch_bar = None
        super().__init__()

    def setup(self, **info):
        """Called once at the beginning of training."""
        self.epoch_bar = tqdm(
            desc="Training", unit="epoch", leave=True, dynamic_ncols=True, total=self.max_epochs
        )

    def handle_result(self, results: Dict, **info):
        """Called when a trial reports a result."""
        if self.epoch_bar is None:
            self.setup()
        
        train_loss = results.get("train_loss", 0.0)
        val_loss = results.get("val_loss", 0.0)

        self.epoch_bar.set_postfix_str(f"train_loss={train_loss:0.3f} | val_loss={val_loss:0.3f} ")
        self.epoch_bar.update()

    def finish(self, error: bool = False, **info):
        """Called at the end of training."""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
