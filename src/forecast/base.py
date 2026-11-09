"""예측기 공통 인터페이스.

모든 예측기는 '모델 스케일로 변환된 전체 시계열 x' 와 origin 위치 배열을 받아
각 origin 에서 h 스텝 뒤 값의 분위수 (n_origins, K) 를 모델 스케일로 돌려준다.
x[i] 는 origin i 시점까지 알려진 trailing 지표값이다 (미래 정보 없음).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    name: str
    uses_context: bool          # context_lengths 를 순회해야 하는지 (Timestral: True)

    def predict(self, x: np.ndarray, origins: np.ndarray, horizon: int,
                ctx: int | None, levels: np.ndarray, key: str = "default") -> np.ndarray:
        """key: (티커, 시계열) 식별자. 차수 캐시 등 상태 분리에 쓰임."""
        ...


def get_forecaster(model: str, exp_cfg: dict) -> "Forecaster":
    """모델 이름 -> 예측기. timestral-{5M|28M|123M|209M} | arima | climatology"""
    if model.startswith("timestral-"):
        from src.forecast.timestral_runner import TimestralRunner
        return TimestralRunner(model.split("-", 1)[1], exp_cfg["timestral"])
    if model == "arima":
        from src.forecast.arima_runner import ArimaRunner
        return ArimaRunner(exp_cfg["arima"])
    if model == "climatology":
        from src.forecast.climatology_runner import ClimatologyRunner
        return ClimatologyRunner(exp_cfg["climatology"])
    raise ValueError(f"unknown model: {model}")
