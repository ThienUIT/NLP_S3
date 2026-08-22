"""Feeds real UTS2017_Bank questions to a running server (docker-compose up),
one every few seconds, to demonstrate the live routing/notification flow --
run against the server, not inside it (uses the project venv/HF access, not
the container's minimal one).

Usage: .venv/Scripts/python.exe server/simulate.py --n 15 --delay 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s3_reproduction.validate_uts_bank import build_label_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bắn câu hỏi mẫu thật vào server đang chạy để demo routing")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, _ = build_label_frame()
    sample = frame.sample(n=min(args.n, len(frame)), random_state=args.seed)

    print(f"Đang bắn {len(sample)} câu hỏi tới {args.url} (mỗi {args.delay}s)...\n")
    correct = 0
    for i, row in enumerate(sample.itertuples(), start=1):
        payload = {"text": row.text, "id": f"sim-{i}"}
        try:
            response = requests.post(f"{args.url}/questions", json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[{i}] Lỗi gọi server: {exc} -- server đã `docker compose up` chưa?")
            return
        result = response.json()
        match = "✓" if result["department"] == row.label else " "
        correct += result["department"] == row.label
        print(
            f"[{i:>2}/{len(sample)}] nhãn thật={row.label:18s} -> định tuyến={result['department']:18s} "
            f"(điểm={result['score']:.2f}) {match}"
        )
        time.sleep(args.delay)

    print(f"\nKhớp đúng nhãn thật: {correct}/{len(sample)} -- đây chỉ là điểm tham khảo nhanh, "
          "xem validate_uts_bank.py --combined để có AUC đầy đủ trên toàn bộ dữ liệu.")


if __name__ == "__main__":
    main()
