"""Timestral 제로샷 예측기 (5M / 28M / 123M / 209M).

API (mistralai/timestral README 기준):
    model = TimestralForecaster.from_pretrained(repo_or_local_dir, device=...)
    fc = model.forecast(context, prediction_length=H)   # context: (B, T) numpy/torch
    fc.quantiles        -> (B, V, H, 21)
    fc.quantile_levels  -> (0.01, 0.05, ..., 0.99)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.common import ROOT

log = logging.getLogger(__name__)


def _to_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):                      # torch.Tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def _step_quantiles(q: np.ndarray, horizon: int) -> np.ndarray:
    """(B, V, H, Q) | (B, H, Q) -> (B, Q): 첫 변량의 h 번째 스텝."""
    if q.ndim == 4:
        q = q[:, 0]
    if q.ndim != 3:
        raise ValueError(f"예상치 못한 quantiles shape: {q.shape}")
    return q[:, horizon - 1, :]


class TimestralRunner:
    uses_context = True

    def __init__(self, size: str, cfg: dict, model=None):
        self.size = size
        self.name = f"timestral-{size}"
        self.cfg = cfg
        self._model = model                       # 테스트에서 가짜 모델 주입 가능
        self.last_ms_per_forecast: float | None = None

    # ---- 로딩 ---------------------------------------------------------
    def _source(self) -> str:
        local = ROOT / self.cfg.get("models_dir", "models") / f"Timestral-{self.size}"
        if (local / "config.json").exists():
            return str(local)
        return f"{self.cfg.get('repo_prefix', 'mistralai/')}Timestral-{self.size}"

    def _load(self):
        if self._model is None:
            from timestral import TimestralForecaster          # pip install git+https://github.com/mistralai/timestral.git
            src = self._source()
            log.info("loading %s from %s (device=%s)", self.name, src, self.cfg.get("device", "cpu"))
            self._model = TimestralForecaster.from_pretrained(src, device=self.cfg.get("device", "cpu"))
        return self._model

    # ---- 예측 ---------------------------------------------------------
    def predict(self, x, origins, horizon, ctx, levels, key="default"):
        model = self._load()
        bs = int(self.cfg.get("batch_size", 32))
        out = np.empty((len(origins), len(levels)))
        t0 = time.perf_counter()
        for s in range(0, len(origins), bs):
            idx = origins[s:s + bs]
            batch = np.stack([x[i + 1 - ctx:i + 1] for i in idx]).astype(np.float32)   # (B, ctx)
            if np.isnan(batch).any():
                raise ValueError("컨텍스트에 NaN 이 있음: origin 이 첫 유효값에서 너무 가깝습니다")
            fc = model.forecast(batch, prediction_length=horizon)
            q = _step_quantiles(_to_numpy(fc.quantiles), horizon)                      # (B, Q)
            model_levels = np.asarray(getattr(fc, "quantile_levels", levels), dtype=float)
            if len(model_levels) != len(levels) or not np.allclose(model_levels, levels, atol=1e-6):
                q = np.stack([np.interp(levels, model_levels, row) for row in q])
            out[s:s + len(idx)] = q
        self.last_ms_per_forecast = (time.perf_counter() - t0) / max(len(origins), 1) * 1000
        return out
