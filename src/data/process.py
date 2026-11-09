"""정제본 -> 목표 시계열(m5, m20, yz20) 계산 -> data/processed/{KEY}.parquet + meta.json"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.common import DATA_INTERIM, DATA_PROCESSED, ensure_dirs, load_config
from src.targets.spec import load_series_specs

log = logging.getLogger(__name__)


def build_processed(keys: list[str] | None = None) -> None:
    ensure_dirs()
    cfg = load_config("data")
    specs = load_series_specs()
    meta = {"series": {n: vars(s) for n, s in specs.items()}, "tickers": {}}
    for key in cfg["tickers"]:
        if keys and key not in keys:
            continue
        ohlc = pd.read_parquet(DATA_INTERIM / f"{key}.parquet")
        out = ohlc.copy()
        for name, spec in specs.items():
            out[name] = spec.compute(ohlc)
        out.to_parquet(DATA_PROCESSED / f"{key}.parquet")
        meta["tickers"][key] = {
            "rows": int(len(out)),
            "start": str(out.index[0].date()),
            "end": str(out.index[-1].date()),
            "first_valid": {n: str(out[n].first_valid_index().date()) for n in specs},
        }
        log.info("%s: processed %d rows -> %s", key, len(out), DATA_PROCESSED / f"{key}.parquet")
    with open(DATA_PROCESSED / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_processed(key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"{key}.parquet")
