"""
Lab 1 coordinator.

Compares numpy vs distributed MapReduce for:
  1. Matrix multiplication  A[N×N] @ B[N×N]
  2. Linear regression      AX = Y  (normal equations via matmul)

Workers are R processes communicating via NATS.

Memory note
-----------
Three N×N float64 matrices require 3 * N^2 * 8 bytes of RAM.
For N=15 000 that is ~5.4 GB — safe on 8 GB free RAM.
For N=18 000 it reaches ~7.8 GB, which risks OOM.
NUMPY_MAX_SIZE is set conservatively at 15 000.
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


def separator(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


async def run_matmul(nc: nats.NATS) -> None:
    separator("Matrix Multiplication: A[N×N] @ B[N×N]")

    N = read_positive_int("Enter matrix size N: ")

    if N > 500:
        print(f"  Note: MapReduce sends N²={N**2:,} tasks — this may be slow for N > 500.")

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
    print(f"[MapReduce] computing {N*N:,} tasks across workers...")
    t0 = time.perf_counter()
    C_mr = await matmul_mapreduce(nc, A, B)
    t_mr = time.perf_counter() - t0
    print(f"[MapReduce] {N}×{N} @ {N}×{N}  →  {t_mr:.4f}s")

    # --- comparison ---
    if C_numpy is not None:
        match = np.allclose(C_numpy, C_mr, atol=1e-8)
        max_diff = float(np.max(np.abs(C_numpy - C_mr)))
        print(f"\nMatch:      {'YES ✓' if match else 'NO ✗'}  (max_diff={max_diff:.2e})")
        ratio = t_np / t_mr
        if ratio < 1:
            print(f"Speedup:    MapReduce is {1/ratio:.1f}x faster than numpy")
        else:
            print(f"Speedup:    numpy is {ratio:.1f}x faster than MapReduce")


async def run_linreg(nc: nats.NATS) -> None:
    separator("Linear Regression: solve AX = Y")
    print("  (Normal equations: X = (A'A)^{-1} A'Y — computed via MapReduce matmul)\n")

    M = read_positive_int("Enter number of samples M: ")
    K = read_positive_int("Enter number of features K: ")

    rng = np.random.default_rng(7)
    A      = rng.standard_normal((M, K))
    X_true = rng.standard_normal((K, 1))
    noise  = rng.standard_normal((M, 1)) * 0.01
    Y      = A @ X_true + noise

    At = A.T  # shape K×M

    # --- numpy ---
    if max(M, K) <= NUMPY_MAX_SIZE:
        t0 = time.perf_counter()
        X_numpy, *_ = np.linalg.lstsq(A, Y, rcond=None)
        t_np = time.perf_counter() - t0
        print(f"[numpy]     lstsq {M}×{K}  →  {t_np:.4f}s")
    else:
        X_numpy = None
        print(f"[numpy]     skipped — matrix exceeds NUMPY_MAX_SIZE={NUMPY_MAX_SIZE}")

    # --- MapReduce ---
    print(f"[MapReduce] step 1/2 — computing A'A ({K}×{K} tasks)...")
    t0 = time.perf_counter()
    AtA = await matmul_mapreduce(nc, At, A)   # K×M @ M×K = K×K

    print(f"[MapReduce] step 2/2 — computing A'Y ({K}×1 tasks)...")
    AtY = await matmul_mapreduce(nc, At, Y)   # K×M @ M×1 = K×1

    X_mr = np.linalg.solve(AtA, AtY)
    t_mr = time.perf_counter() - t0
    print(f"[MapReduce] solve {M}×{K}  →  {t_mr:.4f}s")

    # --- comparison ---
    if X_numpy is not None:
        match = np.allclose(X_numpy, X_mr, atol=1e-6)
        max_diff = float(np.max(np.abs(X_numpy - X_mr)))
        print(f"\nMatch:      {'YES ✓' if match else 'NO ✗'}  (max_diff={max_diff:.2e})")
        ratio = t_np / t_mr
        if ratio < 1:
            print(f"Speedup:    MapReduce is {1/ratio:.1f}x faster than numpy")
        else:
            print(f"Speedup:    numpy is {ratio:.1f}x faster than MapReduce")


async def main() -> None:
    nats_url = os.environ.get("NATS_URL", "nats://nats:4222")
    print(f"Connecting to NATS at {nats_url}...")
    nc = await nats.connect(nats_url)

    # Give workers time to subscribe after coordinator starts
    await asyncio.sleep(1)

    print("\nLab 1 — MapReduce vs numpy benchmark")
    print("Workers are R processes receiving tasks via NATS.\n")

    await run_matmul(nc)
    await run_linreg(nc)

    await nc.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
