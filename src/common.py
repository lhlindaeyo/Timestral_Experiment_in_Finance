"""경로, 설정 로딩, 분위수 레벨 유틸."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FORECASTS = RESULTS / "forecasts"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
LOGS = ROOT / "logs"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """configs/{name}.yaml 로딩 (data | targets | experiment)."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quantile_levels() -> np.ndarray:
    return np.asarray(load_config("targets")["quantile_levels"], dtype=float)


def qcol(tau: float) -> str:
    """분위수 레벨 -> 컬럼명. 0.05 -> 'q05', 0.5 -> 'q50', 0.99 -> 'q99'."""
    return f"q{int(round(tau * 100)):02d}"


def qcols(levels=None) -> list[str]:
    levels = quantile_levels() if levels is None else levels
    return [qcol(t) for t in levels]


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FORECASTS, METRICS, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
