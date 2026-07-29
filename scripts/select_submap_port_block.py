#!/usr/bin/env python3
"""Select deterministic, currently free port blocks for submap lanes."""

from __future__ import annotations

import argparse
import hashlib
import socket


def candidates(seed: str, lanes: int) -> list[int]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % 1000
    width = lanes * 20
    return [
        20000 + width * ((offset + index) % 1000)
        for index in range(1000)
        if 20000 + width * ((offset + index) % 1000) + width < 60000
    ]


def ports_are_free(start: int, lanes: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for lane in range(lanes):
            for port in range(start + lane * 20, start + lane * 20 + 6):
                handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                handle.bind(("127.0.0.1", port))
                sockets.append(handle)
        return True
    except OSError:
        return False
    finally:
        for handle in sockets:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    if args.lanes < 1 or args.lanes > 3:
        parser.error("lanes must be in [1, 3]")
    for start in candidates(args.seed, args.lanes):
        if ports_are_free(start, args.lanes):
            print(start)
            return 0
    raise RuntimeError("no free port block for requested lanes")


if __name__ == "__main__":
    raise SystemExit(main())
