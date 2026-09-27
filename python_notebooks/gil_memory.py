"""
Memory-bound example:
Sequential vs 2 Threads vs 2 Processes

The task performs a very simple operation on large NumPy arrays.

Because the computation per element is tiny, performance is mainly
limited by memory bandwidth rather than arithmetic.

NumPy performs the array operation in compiled code and can release
the Python GIL.
"""

import os
import time
import threading
import multiprocessing
import numpy as np


N = 200_000_000


def memory_work(a, b):
    result = a + b
    return result


def process_work():
    # Create arrays inside each process
    a = np.ones(N)
    b = np.ones(N)

    memory_work(a, b)


def run_sequential(a1, b1, a2, b2):

    start = time.perf_counter()

    memory_work(a1, b1)
    memory_work(a2, b2)

    return time.perf_counter() - start


def run_threads(a1, b1, a2, b2):

    start = time.perf_counter()

    t1 = threading.Thread(
        target=memory_work,
        args=(a1, b1)
    )

    t2 = threading.Thread(
        target=memory_work,
        args=(a2, b2)
    )

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    return time.perf_counter() - start


def run_processes():

    start = time.perf_counter()

    p1 = multiprocessing.Process(
        target=process_work
    )

    p2 = multiprocessing.Process(
        target=process_work
    )

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    return time.perf_counter() - start


if __name__ == "__main__":

    print("Available CPUs:", os.cpu_count())
    print()

    print("Creating arrays for sequential/thread tests...")

    a1 = np.ones(N)
    b1 = np.ones(N)

    a2 = np.ones(N)
    b2 = np.ones(N)

    print("Arrays created\n")

    sequential_time = run_sequential(
        a1, b1, a2, b2
    )

    print(
        f"Sequential  : {sequential_time:.3f} s"
    )

    thread_time = run_threads(
        a1, b1, a2, b2
    )

    print(
        f"2 threads   : {thread_time:.3f} s"
    )

    process_time = run_processes()

    print(
        f"2 processes : {process_time:.3f} s"
    )

    print("\n--- Speedup ---")

    print(
        f"Threads   : "
        f"{sequential_time / thread_time:.2f}x"
    )

    print(
        f"Processes : "
        f"{sequential_time / process_time:.2f}x"
    )