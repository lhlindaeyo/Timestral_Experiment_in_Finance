"""데이터 준비 / 가중치 다운로드용 CLI. 모델 실행·평가·시각화는 notebooks/ 에서 함수를 불러 수행한다.

  python run.py fetch [--tickers KOSPI]      # 1. 데이터 다운로드 (Yahoo Finance) -> data/raw
  python run.py build [--tickers KOSPI]      # 2. 정제 + m5/m20/yz20 계산 -> data/interim, data/processed
  python run.py download-models [--sizes 5M 209M]   # 3. HF 가중치 -> models/
"""
from __future__ import annotations

import argparse
import logging

from src.common import LOGS, ROOT, ensure_dirs, load_config


def cmd_download_models(sizes):
    from huggingface_hub import snapshot_download
    prefix = load_config("experiment")["timestral"]["repo_prefix"]
    for s in sizes:
        target = ROOT / "models" / f"Timestral-{s}"
        print(f"downloading {prefix}Timestral-{s} -> {target}")
        snapshot_download(repo_id=f"{prefix}Timestral-{s}", local_dir=target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fetch").add_argument("--tickers", nargs="*")
    sp.add_parser("build").add_argument("--tickers", nargs="*")
    p = sp.add_parser("download-models")
    p.add_argument("--sizes", nargs="*", default=["5M", "28M", "123M", "209M"])
    a = ap.parse_args()

    ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "run.log", encoding="utf-8")])
    if a.cmd == "fetch":
        from src.data.fetch import fetch_all; fetch_all(a.tickers)
    elif a.cmd == "build":
        from src.data.clean import clean_all
        from src.data.process import build_processed
        clean_all(a.tickers); build_processed(a.tickers)
    elif a.cmd == "download-models":
        cmd_download_models(a.sizes)


if __name__ == "__main__":
    main()
