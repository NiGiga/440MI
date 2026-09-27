"""
The Python GIL (Global Interpreter Lock): Sequential vs Threads vs Processes

The Python GIL is a mechanism in the standard CPython interpreter that allows only one thread at a time to execute Python bytecode.

This means that for CPU-bound Python code, multiple threads usually do not run Python instructions in parallel on multiple cores. For example, two threads doing heavy Python calculations will typically not give a speedup.
The GIL simplifies memory management in CPython, but limits true parallel execution of Python threads for CPU-bound code.

Process: Python program
   ├── Thread 1: read data
   ├── Thread 2: handle a request
   └── Thread 3: perform another task
A thread is a lightweight unit of execution within a process that shares the same memory space with other threads.

----------------

We consider a CPU-bound task, i.e. a task that spends most
of its time performing computations.

We run the same task twice using:

1. sequential execution
2. two threads
3. two processes

For CPU-bound pure Python code, the GIL prevents multiple threads from executing Python bytecode at the same time.

Separate processes, instead, have separate Python interpreters
and separate GILs, so they can run on different CPU cores.
"""

import os
import time
import threading
import multiprocessing

N = 120_000_000


def cpu_work():
    total = 0

    for i in range(N):
        total += (i % 7) * (i % 13)

    return total


def run_sequential():
    start = time.perf_counter()

    cpu_work()
    cpu_work()

    return time.perf_counter() - start


def run_threads():
    start = time.perf_counter()

    t1 = threading.Thread(target=cpu_work)
    t2 = threading.Thread(target=cpu_work)

    t1.start()
    t2.start()

    t1.join()            # wait here until t1 finishes
    t2.join()            # wait here until t2 finishes

    return time.perf_counter() - start


def run_processes():
    start = time.perf_counter()

    p1 = multiprocessing.Process(target=cpu_work)
    p2 = multiprocessing.Process(target=cpu_work)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    return time.perf_counter() - start


if __name__ == "__main__":

    print("Available CPUs:", os.cpu_count())
    print()

    sequential_time = run_sequential()
    print(f"Sequential  : {sequential_time:.2f} s")

    thread_time = run_threads()
    print(f"2 threads   : {thread_time:.2f} s")

    process_time = run_processes()
    print(f"2 processes : {process_time:.2f} s")

    print("\n--- Speedup ---")

    print(
        f"Threads   : {sequential_time / thread_time:.2f}x"
    )

    print(
        f"Processes : {sequential_time / process_time:.2f}x"
    )

'''
SEQUENTIAL

Task 1: ███████████████
Task 2:                ███████████████

time →


2 THREADS

Thread 1: ████    ████    ████
Thread 2:     ████    ████    ████
              ↑
             GIL

time →

The threads alternate execution of Python bytecode.
For CPU-bound pure Python code, we therefore expect
little or no speed-up.


2 PROCESSES

Process 1: ███████████████
Process 2: ███████████████
           ↑
      running in parallel

time →

Each process has its own Python interpreter and its own GIL.
On a multicore machine, the two processes can execute in parallel.

'''