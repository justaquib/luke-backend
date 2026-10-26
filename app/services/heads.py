"""A softmax classifier over the assistant's neural embeddings.

The heads are the small learned parts of the brain: the feeling in a message,
the kind of question that was asked, whether the answer is likely to be right.
They all run on PyTorch, the same library the word model uses, so the app
needs one library for its models and not two.
"""

from __future__ import annotations

from pathlib import Path

import torch


class LinearHead:
    """A softmax classifier over the assistant's neural embeddings."""

    def __init__(self, labels: list[str], seed: int = 11) -> None:
        self.labels = labels
        self.seed = seed
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.fitted = False

    def fit(
        self,
        features: list[list[float]],
        targets: list[int],
        epochs: int = 200,
        learning_rate: float = 0.1,
    ) -> bool:
        if len(features) < max(4, len(self.labels)):
            return False
        if len(set(targets)) < 2:
            # There is only one kind of answer to learn from, so the head
            # learns that: it says the same for everything it was shown.
            x = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.long)
            counts = torch.bincount(y, minlength=len(self.labels)).to(torch.float32)
            self.weights = torch.zeros((x.shape[1], len(self.labels)), dtype=torch.float32)
            self.bias = torch.log(counts.clamp(min=1.0) / counts.sum())
            self.fitted = True
            return True
        x = torch.tensor(features, dtype=torch.float32)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y = torch.tensor(targets, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        self.weights = (
            torch.randn((x.shape[1], len(self.labels)), generator=generator, dtype=torch.float32)
            * 0.05
        )
        self.bias = torch.zeros(len(self.labels), dtype=torch.float32)
        one_hot = torch.zeros((len(y), len(self.labels)), dtype=torch.float32)
        one_hot[torch.arange(len(y)), y] = 1
        # Rare feelings should count as much as common ones.
        counts = torch.bincount(y, minlength=len(self.labels)).to(torch.float32)
        class_weights = counts.sum() / (len(self.labels) * counts.clamp(min=1.0))
        weights = class_weights[y].reshape(-1, 1)

        for _ in range(epochs):
            logits = x @ self.weights + self.bias
            logits -= logits.max(dim=1, keepdim=True).values
            probabilities = torch.exp(logits)
            probabilities /= probabilities.sum(dim=1, keepdim=True)
            grad = torch.clamp((probabilities - one_hot) * weights / len(y), -1.0, 1.0)
            grad = torch.nan_to_num(grad)
            if not torch.isfinite(grad).all():
                # The numbers went out of hand, so the training stops here
                # instead of writing nonsense into the model.
                break
            self.weights -= learning_rate * (x.T @ grad + 0.002 * self.weights)
            self.bias -= learning_rate * grad.sum(dim=0)
        self.weights = torch.nan_to_num(self.weights)
        self.bias = torch.nan_to_num(self.bias)
        self.fitted = True
        return True

    def predict(self, features: list[list[float]]) -> tuple[str, float, dict[str, float]]:
        if not self.fitted or self.weights is None or self.bias is None:
            return self.labels[0], 0.0, {}
        x = torch.tensor(features, dtype=torch.float32)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        logits = x @ self.weights + self.bias
        logits -= logits.max(dim=1, keepdim=True).values
        probabilities = torch.exp(logits)
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        index = int(torch.argmax(probabilities[0]).item())
        scores = {
            label: round(float(value), 3)
            for label, value in zip(self.labels, probabilities[0].tolist(), strict=True)
        }
        return self.labels[index], round(float(probabilities[0][index].item()), 3), scores

    def save(self, path: Path) -> None:
        if self.weights is None or self.bias is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"labels": list(self.labels), "weights": self.weights, "bias": self.bias},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LinearHead | None:
        if not path.is_file():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            head = cls([str(label) for label in data["labels"]])
            head.weights = data["weights"].to(torch.float32)
            head.bias = data["bias"].to(torch.float32)
            head.fitted = True
            return head
        except Exception:
            return None
