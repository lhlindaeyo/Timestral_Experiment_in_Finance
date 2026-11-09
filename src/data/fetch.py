"""Yahoo Finance 에서 지수 일봉 OHLC 다운로드 -> data/raw/{KEY}.parquet"""
from __future__ import annotations

import logging

import pandas as pd

from src.common import DATA_RAW, ensure_dirs, load_config

log = logging.getLogger(__name__)


def fetch_one(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"{symbol}: 데이터가 비어 있음 (네트워크/심볼 확인)")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_all(keys: list[str] | None = None) -> None:
    ensure_dirs()
    cfg = load_config("data")
    for key, t in cfg["tickers"].items():
        if keys and key not in keys:
            continue
        df = fetch_one(t["symbol"], cfg["start"], cfg.get("end"))
        path = DATA_RAW / f"{key}.parquet"
        df.to_parquet(path)
        log.info("%s (%s): %d rows %s ~ %s -> %s", key, t["symbol"], len(df),
                 df.index[0].date(), df.index[-1].date(), path)
