"""
Lab 1, part 1 — Matrix multiplication benchmark: numpy vs MapReduce.

Usage:
    docker compose run --rm coordinator python matmul.py

Prompts for matrix size N, then runs A[N×N] @ B[N×N] both ways and compares.

Memory note
-----------
Three N×N float64 matrices require 3 * N^2 * 8 bytes of RAM.
For N=15 000 that is ~5.4 GB — safe on 8 GB free RAM.
NUMPY_MAX_SIZE is set conservatively at 15 000; numpy is skipped above it.
"""

import asyncio
import os
import time

import numpy as np
import nats

from mapreduce import matmul_mapreduce

NUMPY_MAX_SIZE = 15_000


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt))
            if val > 0:
                return val
            print("  Please enter a positive integer.")
        except ValueError:
            print("  Invalid input, try again.")


async def main() -> None:
    nats_url = os.environ.get("NATS_URL", "nats://nats:4222")
    nc = await nats.connect(nats_url)
    await asyncio.sleep(1)  # let workers finish subscribing

    print("=== Matrix Multiplication: A[N×N] @ B[N×N] ===\n")
    N = read_positive_int("Enter matrix size N: ")

    if N > 500:
        print(f"  Note: MapReduce sends N²={N**2:,} tasks — may be slow for N > 500.")

    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    B = rng.standard_normal((N, N))

    # --- numpy ---
    if N <= NUMPY_MAX_SIZE:
        t0 = time.perf_counter()
        C_numpy = A @ B
        t_np = time.perf_counter() - t0
        print(f"\n[numpy]     {N}×{N} @ {N}×{N}  →  {t_np:.4f}s")
    else:
        C_numpy = None
        print(f"\n[numpy]     skipped — N={N} exceeds NUMPY_MAX_SIZE={NUMPY_MAX_SIZE}")

    # --- MapReduce ---
    print(f"[MapReduce] distributing {N*N:,} tasks to R workers...")
    t0 = time.perf_counter()
    C_mr = await matmul_mapreduce(nc, A, B)
    t_mr = time.perf_counter() - t0
    print(f"[MapReduce] {N}×{N} @ {N}×{N}  →  {t_mr:.4f}s")

    # --- comparison ---
    if C_numpy is not None:
        match = np.allclose(C_numpy, C_mr, atol=1e-8)
        max_diff = float(np.max(np.abs(C_numpy - C_mr)))
        print(f"\nMatch:    {'YES ✓' if match else 'NO ✗'}  (max_diff={max_diff:.2e})")
        ratio = t_mr / t_np  # how many times slower MapReduce is vs numpy
        if ratio >= 1:
            print(f"Speedup:  numpy is {ratio:.1f}x faster than MapReduce")
        else:
            print(f"Speedup:  MapReduce is {1/ratio:.1f}x faster than numpy")

    await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
