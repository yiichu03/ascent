#!/usr/bin/env python3
"""Fail-closed health probe for ASCENT's six local model services."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import requests


CHECKS = (
    ("qwen2_5", "QWEN2_5_PORT", "/qwen2_5"),
    ("blip2itm", "BLIP2ITM_PORT", "/blip2itm"),
    ("sam", "SAM_PORT", "/mobile_sam"),
    ("gdino", "GROUNDING_DINO_PORT", "/gdino"),
    ("ram", "RAM_PORT", "/ram"),
    ("dfine", "DFINE_PORT", "/dfine"),
)


def probe_services(
    *, request_timeout: float
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for name, port_key, route in CHECKS:
        port = os.environ.get(port_key)
        if not port:
            results.append(
                {
                    "name": name,
                    "port_key": port_key,
                    "ok": False,
                    "error": "missing_port_environment",
                }
            )
            continue
        url = f"http://127.0.0.1:{port}{route}"
        try:
            response = requests.get(url, timeout=request_timeout)
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status_code": response.status_code,
                    "ok": response.status_code == 200,
                    "response_prefix": response.text[:160],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:240],
                }
            )
    return results


def wait_for_services(
    *,
    stage: str,
    wait_seconds: float,
    interval_seconds: float,
    request_timeout: float,
) -> int:
    deadline = time.monotonic() + wait_seconds
    attempt = 0
    while True:
        attempt += 1
        results = probe_services(request_timeout=request_timeout)
        print(
            json.dumps(
                {"stage": stage, "attempt": attempt, "services": results},
                sort_keys=True,
            ),
            flush=True,
        )
        if all(bool(result["ok"]) for result in results):
            print(
                f"service_health=PASS stage={stage} attempts={attempt}",
                flush=True,
            )
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"service_health=FAIL stage={stage} attempts={attempt}",
                flush=True,
            )
            return 24
        time.sleep(min(interval_seconds, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    args = parser.parse_args()
    if (
        args.wait_seconds < 0
        or args.interval_seconds <= 0
        or args.request_timeout <= 0
    ):
        parser.error(
            "wait must be non-negative; interval and timeout positive"
        )
    return wait_for_services(
        stage=args.stage,
        wait_seconds=args.wait_seconds,
        interval_seconds=args.interval_seconds,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
