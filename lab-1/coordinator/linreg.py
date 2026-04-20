"""
Lab 1, part 2 — Linear regression benchmark: numpy vs MapReduce.

Usage:
    docker compose run --rm coordinator python linreg.py

Solves AX = Y via normal equations: X = (A'A)^{-1} A'Y.
A'A and A'Y are computed with the MapReduce matmul from mapreduce.py.
Final solve (small KxK system) runs in the coordinator with numpy.

Prompts for number of samples M and features K.
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
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input, try again.")


async def main() -> None:
    nats_url = os.environ.get("NATS_URL", "nats://nats:4222")
    nc = await nats.connect(nats_url)
    await asyncio.sleep(1)  # let workers finish subscribing

    print("Linear Regression:\tsolve AX = Y")
    print("Method:\t\tnormal equations X = (A'A)^{-1} A'Y")

    M = read_positive_int("Enter number of samples M: ")
    K = read_positive_int("Enter number of features K: ")

    rng    = np.random.default_rng(7)
    A      = rng.standard_normal((M, K))
    X_true = rng.standard_normal((K, 1))
    Y      = A @ X_true + rng.standard_normal((M, 1)) * 0.01

    At = A.T  # shape K×M

    # --- numpy ---
    if max(M, K) <= NUMPY_MAX_SIZE:
        t0 = time.perf_counter()
        X_numpy, *_ = np.linalg.lstsq(A, Y, rcond=None)
        t_np = time.perf_counter() - t0
        print(f"[numpy]\t\tlstsq {M}×{K}  →  {t_np:.4f}s")
    else:
        X_numpy = None
        print(f"[numpy]\t\tskipped — matrix exceeds NUMPY_MAX_SIZE={NUMPY_MAX_SIZE}")

    # --- MapReduce ---
    print(f"[MapReduce]\tstep 1/2 — A'A  ({K}x{K} output, {K*M:,} tasks)...")
    t0 = time.perf_counter()
    AtA = await matmul_mapreduce(nc, At, A)   # K×M @ M×K = K×K

    print(f"[MapReduce]\tstep 2/2 — A'Y  ({K}x1 output, {K:,} tasks)...")
    AtY = await matmul_mapreduce(nc, At, Y)   # KxM @ Mx1 = Kx1

    X_mr = np.linalg.solve(AtA, AtY)
    t_mr = time.perf_counter() - t0
    print(f"[MapReduce]\tdone\t\t{t_mr:.4f}s")

    # --- comparison ---
    if X_numpy is not None:
        match = np.allclose(X_numpy, X_mr, atol=1e-6)
        max_diff = float(np.max(np.abs(X_numpy - X_mr)))
        print(f"Match:\t\t{'YES' if match else 'NO'}\t\tmax_diff={max_diff:.2e}")
        ratio = t_mr / t_np  # how many times slower MapReduce is vs numpy
        if ratio >= 1:
            print(f"Speedup:\tnumpy is {ratio:.1f}x faster than MapReduce")
        else:
            print(f"Speedup:\tMapReduce is {1/ratio:.1f}x faster than numpy")

    await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
