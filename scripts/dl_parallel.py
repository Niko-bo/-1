"""多线程分段下载（用于 download.pytorch.org 这类单连接被限速的源）。

用法：
  python scripts/dl_parallel.py <url> <outfile> [--conn 8]
  python scripts/dl_parallel.py --bench <url> [--conn 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

CHUNK = 1024 * 512


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def head(url: str) -> tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "dsh-dl/1.0"})
    with opener().open(req, timeout=30) as r:
        size = int(r.headers.get("Content-Length") or 0)
        ranges = (r.headers.get("Accept-Ranges") or "").lower() == "bytes"
    return size, ranges


def worker(url, start, end, path, idx, results, lock):
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-dl/1.0",
                                               "Range": f"bytes={start}-{end}"})
    got = 0
    try:
        with opener().open(req, timeout=60) as r, open(path, "r+b") as f:
            f.seek(start)
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                with lock:
                    results[idx] = got
    except Exception as e:  # noqa: BLE001
        with lock:
            results[idx] = -1
        print(f"  conn{idx} FAIL {e}", flush=True)


def download(url: str, dest: Path, conn: int = 8) -> bool:
    size, ranges = head(url)
    if size <= 0:
        print("cannot determine size")
        return False
    if not ranges:
        print("server does not support ranges, single stream")
        conn = 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.truncate(size)

    parts = [(i * size // conn, (i + 1) * size // conn - 1) for i in range(conn)]
    results = [0] * conn
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(url, a, b, dest, i, results, lock), daemon=True)
               for i, (a, b) in enumerate(parts)]
    t0 = time.time()
    for t in threads:
        t.start()
    last = 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        now = time.time()
        if now - last > 5:
            last = now
            done = sum(max(0, r) for r in results)
            speed = done / max(1e-6, now - t0) / 1024 ** 2
            print(f"  {done/1024**3:.2f}/{size/1024**3:.2f}GB {speed:.1f}MB/s", flush=True)
    done = sum(max(0, r) for r in results)
    ok = done == size
    print(f"  {'OK' if ok else 'INCOMPLETE'} {done/1024**3:.2f}/{size/1024**3:.2f}GB "
          f"in {time.time()-t0:.0f}s", flush=True)
    return ok


def bench(url: str, conn: int, seconds: float = 8.0) -> None:
    size, _ = head(url)
    span = min(size, 6 * 1024 * 1024)
    got = [0] * conn
    lock = threading.Lock()

    def w(i):
        start = i * span // conn
        end = (i + 1) * span // conn - 1
        req = urllib.request.Request(url, headers={"User-Agent": "dsh-dl/1.0",
                                                   "Range": f"bytes={start}-{end}"})
        t0 = time.time()
        try:
            with opener().open(req, timeout=30) as r:
                while time.time() - t0 < seconds:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    with lock:
                        got[i] += len(buf)
        except Exception:  # noqa: BLE001
            pass

    ts = [threading.Thread(target=w, args=(i,), daemon=True) for i in range(conn)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    total = sum(got)
    print(f"conn={conn}: {total/1024**2:.1f}MB in {time.time()-t0:.1f}s = "
          f"{total/1024**2/(time.time()-t0):.2f}MB/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--conn", type=int, default=8)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench or not args.out:
        bench(args.url, args.conn)
        return 0
    return 0 if download(args.url, Path(args.out), args.conn) else 1


if __name__ == "__main__":
    sys.exit(main())
