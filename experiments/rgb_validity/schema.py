"""Data contracts for frozen prediction batches. Validation only -- nothing is modified.

The methodology this implements analyses predictions that already exist. Every function
here is read-only by construction: it raises on a contract violation rather than
repairing one, because silently coercing a malformed array is how an invalid result
acquires a confident-looking number.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


class ContractError(ValueError):
    """A frozen artifact failed its declared contract."""


@dataclass
class FrozenPredictionBatch:
    """Reference, prediction and baseline for one (mode, group) cell.

    source_id is REQUIRED. Row position within an export is not identity: two modes can
    agree on row 7 while row 7 denotes different images, if upstream ordering or
    exclusions differ. Everything downstream groups and splits on source_id.
    """
    mode: str
    group: str
    truth: np.ndarray                    # (N,3,H,W) float in [0,1]
    prediction: np.ndarray
    prior_prediction: np.ndarray | None
    source_id: np.ndarray                # (N,) original capture-set row
    scene_id: np.ndarray                 # (N,) independent-unit id
    session_id: np.ndarray | None        # (N,) or None when unknown -- never invented
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self):
        n = len(self.truth)
        for name in ("truth", "prediction"):
            a = getattr(self, name)
            if a.ndim != 4 or a.shape[1] != 3:
                raise ContractError(f"{name}: expected (N,3,H,W), got {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ContractError(f"{name}: contains non-finite values")
        if self.prediction.shape != self.truth.shape:
            raise ContractError(
                f"shape mismatch: prediction {self.prediction.shape} vs "
                f"truth {self.truth.shape}")
        if self.prior_prediction is not None:
            if self.prior_prediction.shape != self.truth.shape:
                raise ContractError("prior_prediction shape mismatch")
            if not np.all(np.isfinite(self.prior_prediction)):
                raise ContractError("prior_prediction: non-finite values")
        # Range is checked on truth strictly; predictions may exceed [0,1] before
        # clipping and we want that visible rather than silently squashed.
        if self.truth.min() < -1e-6 or self.truth.max() > 1 + 1e-6:
            raise ContractError(
                f"truth outside [0,1]: [{self.truth.min():.3f}, {self.truth.max():.3f}]")
        for name in ("source_id", "scene_id"):
            a = getattr(self, name)
            if a is None or len(a) != n:
                raise ContractError(f"{name}: expected length {n}")
        if self.session_id is not None and len(self.session_id) != n:
            raise ContractError("session_id: length mismatch")
        return self

    @property
    def n_clipped(self) -> int:
        """Predictions outside [0,1]. Reported, not hidden -- see the guardrails."""
        p = self.prediction
        return int(np.sum((p < 0) | (p > 1)))

    def hashes(self) -> dict:
        return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
                for k, v in (("truth", self.truth), ("prediction", self.prediction),
                             ("prior", self.prior_prediction))
                if v is not None}
