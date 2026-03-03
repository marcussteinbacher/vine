"""
Runner — concurrent, resumable numpy processing.

- ProcessPoolExecutor for CPU-bound parallelism (bypasses GIL)
- Calculation function passed directly — no registry, works with functools.partial
- Spawn-safe (Python 3.14+, Windows): function delivered to workers via initializer
- Per-window results: tuple[obj, float, float] per window
- ScalarWriter  : saves (window_id, scalar_a, scalar_b) for every window
- ObjectWriter  : saves obj via pickle every ``object_stride`` windows
- Resumption at chunk level
- Inner + outer tqdm progress bars
"""

import pickle
import multiprocessing as mp
import numpy as np
import queue
import threading
import tempfile
import time
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Callable

logger = logging.getLogger(__name__)

DTYPE = np.float32


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Module-level global — set once per worker process by _worker_init
_calc_fn_global: Callable | None = None

def _worker_init(calc_fn: Callable) -> None:
    """
    Runs once in each worker process before any tasks are submitted.
    Stores the callable in a module-level global so _worker can call it
    without receiving it on every task (avoids repeated pickling overhead).
    """
    global _calc_fn_global
    _calc_fn_global = calc_fn


def _worker(
    chunk_id: int,
    data_path: str,
    s: int,
    e: int,
    progress_queue: mp.Queue,
) -> tuple[int, list[tuple[object, float, float]]]:
    """
    Applies _calc_fn_global to each (250, 50) window in the chunk individually.
    Reports progress to the main process via progress_queue.
    Returns (chunk_id, [(obj, scalar_a, scalar_b), ...]) — one tuple per window.
    """
    data = np.load(data_path, mmap_mode="r")
    chunk = data[s:e].astype(DTYPE)

    assert callable(_calc_fn_global), "_calc_fn_global not set!"

    results = []
    for window in chunk:                        # each window: (250, 50)
        obj, scalar_a, scalar_b = _calc_fn_global(window)
        results.append((obj, float(scalar_a), float(scalar_b)))
        progress_queue.put(1)                   # signal one window done

    return chunk_id, results


# ---------------------------------------------------------------------------
# Scalar writer — every window, two floats
# ---------------------------------------------------------------------------

class ScalarWriter(threading.Thread):
    """
    Writes scalars_NNNNNN.npy files — one structured array per flush batch,
    with columns: window_id (int64), scalar_a (float32), scalar_b (float32).
    Flushes when flush_threshold items accumulate or stop() is called.
    """

    def __init__(self, temp_dir: Path, flush_threshold: int = 10):
        super().__init__(daemon=True, name="ScalarWriter")
        self.temp_dir = temp_dir
        self.flush_threshold = flush_threshold
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._cond = threading.Condition()
        self._flush_counter = 0

    def enqueue(self, window_id: int, scalar_a: float, scalar_b: float):
        self._queue.put((window_id, scalar_a, scalar_b))
        if self._queue.qsize() >= self.flush_threshold:
            with self._cond:
                self._cond.notify()

    def stop(self, wait: bool = True):
        self._stop_event.set()
        with self._cond:
            self._cond.notify()
        if wait:
            self.join()

    def run(self):
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._queue.qsize() >= self.flush_threshold
                    or self._stop_event.is_set()
                )
            self._flush()
            if self._stop_event.is_set() and self._queue.empty():
                break

    def _flush(self):
        pending: dict[int, tuple[float, float]] = {}
        try:
            while True:
                window_id, a, b = self._queue.get_nowait()
                pending[window_id] = (float(a), float(b))
        except queue.Empty:
            pass

        if not pending:
            return

        rows = np.array(
            [(wid, a, b) for wid, (a, b) in pending.items()],
            dtype=[("window_id", np.int64), ("scalar_a", DTYPE), ("scalar_b", DTYPE)],
        )
        path = self.temp_dir / f"scalars_{self._flush_counter:06d}.npy"
        self._flush_counter += 1
        np.save(path, rows)
        logger.debug("ScalarWriter: %d entries -> %s", len(rows), path)
        logger.info("ScalarWriter flushed %d entries.", len(pending))


# ---------------------------------------------------------------------------
# Object writer — every N-th window only
# ---------------------------------------------------------------------------

class ObjectWriter(threading.Thread):
    """
    Saves objects via pickle only for windows where window_id % stride == 0.
    All other windows are dropped silently at enqueue time.
    Flushes when flush_threshold qualifying items accumulate or stop() is called.
    """

    def __init__(self, temp_dir: Path, stride: int = 250, flush_threshold: int = 4):
        super().__init__(daemon=True, name="ObjectWriter")
        self.temp_dir = temp_dir
        self.stride = stride
        self.flush_threshold = flush_threshold
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._cond = threading.Condition()

    def enqueue(self, window_id: int, obj: object):
        if window_id % self.stride != 0:
            return
        self._queue.put((window_id, obj))
        if self._queue.qsize() >= self.flush_threshold:
            with self._cond:
                self._cond.notify()

    def stop(self, wait: bool = True):
        self._stop_event.set()
        with self._cond:
            self._cond.notify()
        if wait:
            self.join()

    def run(self):
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._queue.qsize() >= self.flush_threshold
                    or self._stop_event.is_set()
                )
            self._flush()
            if self._stop_event.is_set() and self._queue.empty():
                break

    def _flush(self):
        pending: dict[int, object] = {}
        try:
            while True:
                window_id, obj = self._queue.get_nowait()
                pending[window_id] = obj
        except queue.Empty:
            pass

        for window_id, obj in pending.items():
            path = self.temp_dir / f"object_{window_id:06d}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            logger.debug("ObjectWriter: window %d -> %s", window_id, path)

        if pending:
            logger.info("ObjectWriter flushed %d object(s).", len(pending))


# ---------------------------------------------------------------------------
# Inner progress bar listener
# ---------------------------------------------------------------------------

def _inner_bar_listener(
    progress_queue: mp.Queue,
    total_windows: int,
    stop_event: threading.Event,
) -> None:
    """Runs in a thread in the main process, drains the queue into a tqdm bar."""
    with tqdm(total=total_windows, desc="  Windows", unit="win",
              position=1, leave=False) as pbar:
        while not stop_event.is_set() or not progress_queue.empty():
            try:
                n = progress_queue.get(timeout=0.05)
                pbar.update(n)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """
    Applies a per-window calculation across a large (N, 250, 50) array in chunks,
    using multiple CPU cores via ProcessPoolExecutor.

    The calculation function receives a single (250, 50) window and must return:
        tuple[obj, float, float]
    where obj is any picklable object, and the two floats are scalar metrics.

    Scalars are saved for every window. Objects are saved every ``object_stride``
    windows. Resumption is at chunk granularity.

    Parameters
    ----------
    calculation_fn : Callable
        Function or functools.partial with signature: fn(window) -> (obj, float, float).
        Must be a module-level function or a partial wrapping one (picklable).
    data : np.ndarray | None
        Input array. Pass either this or ``data_path``, not both.
    data_path : str | Path | None
        Path to an existing .npy file (workers mmap it, nothing is re-saved).
    chunk_size : int
        Number of windows per task. Default os.cpu_count() * 4.
    object_stride : int
        Save obj every this many windows. Default 250.
    n_workers : int | None
        Worker processes. Defaults to os.cpu_count().
    max_in_flight : int | None
        Backpressure cap on submitted futures. Defaults to n_workers * 4.
    temp_dir : str | Path | None
        Where to write checkpoint files. Defaults to temp/.
    flush_threshold : int | None
        Windows to accumulate before flushing scalars. Defaults to n_workers * 2.
    object_flush_threshold : int
        Objects to accumulate before flushing. Default 4.
    """

    def __init__(
        self,
        calculation_fn: Callable,
        data: np.ndarray | None = None,
        data_path: str | Path | None = None,
        chunk_size: int | None = None,
        object_stride: int = 250,
        n_workers: int | None = None,
        max_in_flight: int | None = None,
        temp_dir: str | Path | None = None,
        flush_threshold: int | None = None,
        object_flush_threshold: int = 4,
    ):
        if data is None and data_path is None:
            raise ValueError("Provide either `data` or `data_path`.")
        if data is not None and data_path is not None:
            raise ValueError("Provide either `data` or `data_path`, not both.")

        self._calc_fn      = calculation_fn
        self.chunk_size    = chunk_size or os.cpu_count() * 4
        self.object_stride = object_stride
        self.n_workers     = n_workers or os.cpu_count()
        self.max_in_flight = max_in_flight or self.n_workers * 4

        if temp_dir is None:
            temp_dir = Path("temp")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if data_path is not None:
            self._data_path   = str(data_path)
            probe             = np.load(self._data_path, mmap_mode="r")
            self._data_len    = probe.shape[0]
        else:
            self._data_path = str(self.temp_dir / "input_data.npy")
            if not Path(self._data_path).exists():
                logger.info("Saving input (%s, float32) to %s ...", data.shape, self._data_path)
                np.save(self._data_path, data.astype(DTYPE))
            self._data_len = data.shape[0]

        self.n_chunks = int(np.ceil(self._data_len / chunk_size))

        _threshold = flush_threshold or self.n_workers * 2
        self._scalar_writer = ScalarWriter(self.temp_dir, flush_threshold=_threshold)
        self._object_writer = ObjectWriter(self.temp_dir,stride=object_stride,flush_threshold=object_flush_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute concurrently. Saves scalars for every window and objects every
        ``object_stride`` windows. Resumes from last completed chunk.
        """
        completed = self._find_completed_chunks()
        remaining = [i for i in range(self.n_chunks) if i not in completed]

        logger.info(
            "Chunks total=%d  done=%d  remaining=%d  workers=%d",
            self.n_chunks, len(completed), len(remaining), self.n_workers,
        )

        if not remaining:
            logger.info("All chunks already computed.")
            return

        total_windows = len(remaining) * self.chunk_size
        progress_queue: mp.Queue = mp.Queue()
        stop_inner = threading.Event()
        inner_listener = threading.Thread(
            target=_inner_bar_listener,
            args=(progress_queue, total_windows, stop_inner),
            daemon=True,
        )

        self._scalar_writer.start()
        self._object_writer.start()
        inner_listener.start()

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers,initializer=_worker_init,initargs=(self._calc_fn)) as pool:
                with tqdm(total=self.n_chunks, initial=len(completed), desc="Chunks", unit="chunk", position=0) as pbar:

                    pending: dict = {}
                    todo = iter(remaining)
                    exhausted = False

                    def _fill():
                        nonlocal exhausted
                        while not exhausted and len(pending) < self.max_in_flight:
                            try:
                                chunk_id = next(todo)
                            except StopIteration:
                                exhausted = True
                                break
                            s, e = self._slice(chunk_id)
                            future = pool.submit(_worker, chunk_id, self._data_path, s, e, progress_queue)
                            pending[future] = chunk_id

                    _fill()

                    while pending:
                        done = [f for f in list(pending) if f.done()]
                        if not done:
                            time.sleep(0.01)
                            continue

                        for future in done:
                            pending.pop(future)
                            chunk_id_out, window_results = future.result()

                            for window_idx, (obj, scalar_a, scalar_b) in enumerate(window_results):
                                global_window_id = chunk_id_out * self.chunk_size + window_idx
                                self._scalar_writer.enqueue(global_window_id, scalar_a, scalar_b)
                                self._object_writer.enqueue(global_window_id, obj)

                            pbar.update(1)

                        _fill()

        finally:
            stop_inner.set()
            inner_listener.join()
            self._scalar_writer.stop(wait=True)
            self._object_writer.stop(wait=True)

    def collect_scalars(self) -> dict[int, tuple[float, float]]:
        """
        Load and merge all scalar checkpoint files.

        Returns
        -------
        dict mapping window_id -> (scalar_a, scalar_b), sorted by window_id. {window_id:(scalar_a, scalar_b), ...}

        Raises
        ------
        RuntimeError if any windows are missing.
        """
        all_files = sorted(self.temp_dir.glob("scalars_*.npy"))
        if not all_files:
            raise RuntimeError("No scalar checkpoints found. Did run() complete?")

        raw: dict[int, tuple[float, float]] = {}
        for path in all_files:
            for row in np.load(path):
                raw[int(row["window_id"])] = (float(row["scalar_a"]), float(row["scalar_b"]))

        total_windows = self.n_chunks * self.chunk_size
        missing = set(range(total_windows)) - raw.keys()
        if missing:
            raise RuntimeError(
                f"{len(missing)} window(s) missing scalars: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

        return dict(sorted(raw.items()))

    def collect_objects(self) -> dict[int, object]:
        """
        Load all saved object checkpoints.

        Returns
        -------
        dict mapping window_id -> obj for every saved stride-boundary window. {window_id: obj}
        """
        result = {}
        for path in sorted(self.temp_dir.glob("object_*.pkl")):
            window_id = int(path.stem.split("_")[1])
            with open(path, "rb") as f:
                result[window_id] = pickle.load(f)
        return result

    def cleanup(self) -> None:
        """Delete all temporary files written by this runner."""
        for pattern in ("scalars_*.npy", "object_*.pkl", "input_data.npy"):
            for p in self.temp_dir.glob(pattern):
                p.unlink(missing_ok=True)
        logger.info("Temporary files removed from %s", self.temp_dir)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _slice(self, chunk_id: int) -> tuple[int, int]:
        s = chunk_id * self.chunk_size
        e = min(s + self.chunk_size, self._data_len)
        return s, e

    def _find_completed_chunks(self) -> set[int]:
        """
        A chunk is complete if its first window's scalar entry exists.
        First window of chunk N has global_window_id = N * chunk_size.
        """
        present_window_ids: set[int] = set()
        for path in self.temp_dir.glob("scalars_*.npy"):
            try:
                for row in np.load(path):
                    present_window_ids.add(int(row["window_id"]))
            except Exception:
                pass

        completed = set()
        for chunk_id in range(self.n_chunks):
            if chunk_id * self.chunk_size in present_window_ids:
                completed.add(chunk_id)
        return completed