"""시계열 정의(SeriesSpec): 지표 계산, 모델 입력 변환, 모델 출력 분위수 -> 원 스케일 역변환.

MDD 는 음수(-12%)로 정의하지만 모델에는 낙폭 크기 m = -mdd (양수) 에 log1p 를 씌워 넣는다.
  * m 은 단조증가 변환이므로 분위수가 그대로 이동한다.
  * 단, 부호가 뒤집히므로 "모델 분위수 레벨 τ'" == "MDD 분위수 레벨 1-τ'" 이다.
    (MDD 하위 5% = 가장 깊은 낙폭 = 낙폭 크기 m 의 상위 95%)
분위수 격자가 0.5 를 중심으로 대칭이라(0.01..0.99) 레벨 뒤집기는 배열 뒤집기로 처리한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import load_config, quantile_levels
from src.targets.mdd import trailing_mdd_series
from src.targets.yang_zhang import yang_zhang

_VOL_FLOOR = 1e-6


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    kind: str                 # "mdd" | "yang_zhang"
    window: int
    transform: str            # "log1p_mag" | "log"
    annualize: int = 252

    # ---- 지표 계산 (라벨 + 입력 공통) -----------------------------------
    def compute(self, ohlc: pd.DataFrame) -> pd.Series:
        if self.kind == "mdd":
            s = trailing_mdd_series(ohlc["close"], self.window)
        elif self.kind == "yang_zhang":
            s = yang_zhang(ohlc, self.window, self.annualize)
        else:
            raise ValueError(f"unknown kind: {self.kind}")
        s.name = self.name
        return s

    # ---- 변환 ---------------------------------------------------------
    @property
    def flips_tail(self) -> bool:
        return self.transform == "log1p_mag"

    def to_model(self, raw) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.transform == "log1p_mag":
            return np.log1p(np.clip(-raw, 0.0, 1.0 - 1e-9))
        if self.transform == "log":
            return np.log(np.clip(raw, _VOL_FLOOR, None))
        raise ValueError(self.transform)

    def quantiles_to_raw(self, q_model: np.ndarray) -> np.ndarray:
        """q_model: (n_origins, K) 모델 스케일 분위수 (표준 레벨 격자 기준) -> 원 스케일, 열 오름차순."""
        q = np.asarray(q_model, dtype=float)
        if self.transform == "log1p_mag":
            levels = quantile_levels()
            assert np.allclose(levels + levels[::-1], 1.0), "분위수 격자가 대칭이어야 함"
            mag = np.clip(np.expm1(q), 0.0, 1.0)
            raw = -mag[:, ::-1]                     # 레벨 뒤집기
        elif self.transform == "log":
            raw = np.exp(q)
        else:
            raise ValueError(self.transform)
        return np.sort(raw, axis=1)                 # 분위수 교차(crossing) 방지


def load_series_specs() -> dict[str, SeriesSpec]:
    cfg = load_config("targets")["series"]
    return {
        name: SeriesSpec(name=name, kind=c["kind"], window=int(c["window"]),
                         transform=c["transform"], annualize=int(c.get("annualize", 252)))
        for name, c in cfg.items()
    }


def load_experiments() -> list[tuple[str, int]]:
    return [(e["series"], int(e["horizon"])) for e in load_config("targets")["experiments"]]
