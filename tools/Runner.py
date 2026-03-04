"""
Runner — concurrent, resumable per-window numpy processing.

- ProcessPoolExecutor for CPU-bound parallelism (bypasses GIL)
- One future per window — eliminates straggler / tail idle time
- Calculation function passed directly, works with functools.partial
- Spawn-safe: calc_fn delivered via initializer
- ScalarWriter  : saves (window_id, scalar_a, scalar_b) for every window
- ObjectWriter  : saves obj via pickle every ``object_stride`` windows
- Resumption at window granularity
- Inner + outer tqdm progress bars
"""

import pickle
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

logger = logging.getLogger(__name__)

DTYPE = np.float32


# ---------------------------------------------------------------------------
# Worker — module-level globals set once per process by _worker_init
# ---------------------------------------------------------------------------

_calc_fn_global        = None   # the calculation callable


def _worker_init(calc_fn):
    """
    Runs once per worker process at startup (via ProcessPoolExecutor initializer).
    Stores calc_fn in a module-level global so it is available in every
    _worker call without pickling on each task submission.
    """
    global _calc_fn_global
    _calc_fn_global = calc_fn


def _worker(window_id: int, data_path: str) -> tuple:
    """
    Process a single window (250 × 50).
    Loads the array via mmap, slices window_id, applies _calc_fn_global.
    Returns (window_id, obj, scalar_a, scalar_b).
    """
    data   = np.load(data_path, mmap_mode="r")
    window = data[window_id].astype(DTYPE)          # shape (250, 50)
    obj, scalar_a, scalar_b = _calc_fn_global(window)
    return window_id, obj, float(scalar_a), float(scalar_b)


# ---------------------------------------------------------------------------
# Scalar writer — every window, two floats
# ---------------------------------------------------------------------------

class ScalarWriter(threading.Thread):
    """
    Daemon thread. Writes scalars_NNNNNN.npy files — one structured array
    per flush batch with columns: window_id (int64), scalar_a (f32), scalar_b (f32).
    Flushes when flush_threshold items accumulate or stop() is called.
    """

    def __init__(self, temp_dir: Path, flush_threshold: int = 10):
        super().__init__(daemon=True, name="ScalarWriter")
        self.temp_dir        = temp_dir
        self.flush_threshold = flush_threshold
        self._queue          = queue.Queue()
        self._stop_event     = threading.Event()
        self._cond           = threading.Condition()
        self._flush_counter  = 0

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
        pending = {}
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
        logger.info("ScalarWriter flushed %d entries -> %s", len(pending), path.name)


# ---------------------------------------------------------------------------
# Object writer — every Nth window only
# ---------------------------------------------------------------------------

class ObjectWriter(threading.Thread):
    """
    Daemon thread. Saves objects via pickle only when window_id % stride == 0.
    All other windows are dropped silently at enqueue() before entering the queue.
    Flushes when flush_threshold qualifying items accumulate or stop() is called.
    """

    def __init__(self, temp_dir: Path, stride: int = 250, flush_threshold: int = 4):
        super().__init__(daemon=True, name="ObjectWriter")
        self.temp_dir        = temp_dir
        self.stride          = stride
        self.flush_threshold = flush_threshold
        self._queue          = queue.Queue()
        self._stop_event     = threading.Event()
        self._cond           = threading.Condition()

    def enqueue(self, window_id: int, obj):
        if self.stride == -1 or window_id % self.stride != 0:
            return                              # not a stride boundary — drop
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
        pending = {}
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

        if pending:
            logger.info("ObjectWriter flushed %d object(s).", len(pending))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """
    Applies a per-window calculation across a large (N, H, W) array,
    submitting one future per window so all workers stay saturated until
    the very last window (no straggler / tail idle time).

    The calculation function receives a single (H, W) window and must return:
        tuple[obj, float, float]
    where obj is any picklable object and the two floats are scalar metrics.

    Scalars  : saved for every window  → scalars_NNNNNN.npy
    Objects  : saved every object_stride windows → object_NNNNNN.pkl
    Resumption: at window granularity — already-computed windows are skipped.

    Parameters
    ----------
    calculation_fn : callable
        fn(window: np.ndarray) -> (obj, scalar_a, scalar_b).
        Must be a module-level function or partial wrapping one (picklable).
    data : np.ndarray | None
        Input array. Pass either this or data_path, not both.
    data_path : str | Path | None
        Path to an existing .npy file (workers mmap it, nothing re-saved).
    object_stride : int
        Save obj every this many windows. Default 250.
    n_workers : int | None
        Worker processes. Defaults to os.cpu_count().
    max_in_flight : int | None
        Max futures in flight (backpressure). Defaults to n_workers * 8.
        Higher values keep workers saturated; lower values reduce memory use.
    temp_dir : str | Path | None
        Where checkpoint files are written.
    flush_threshold : int | None
        Windows to accumulate before flushing scalars. Defaults to n_workers * 4.
    object_flush_threshold : int
        Objects to accumulate before flushing. Default 4.
    """

    def __init__(
        self,
        calculation_fn,
        data=None,
        data_path=None,
        object_stride: int = 250,
        n_workers=None,
        max_in_flight=None,
        temp_dir=None,
        flush_threshold=None,
        object_flush_threshold: int = 4,
    ):
        if data is None and data_path is None:
            raise ValueError("Provide either `data` or `data_path`.")
        if data is not None and data_path is not None:
            raise ValueError("Provide either `data` or `data_path`, not both.")

        self._calc_fn      = calculation_fn
        self.object_stride = object_stride
        self.n_workers     = n_workers or os.cpu_count() or 4
        self.max_in_flight = max_in_flight or self.n_workers * 8

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "runner_checkpoints"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if data_path is not None:
            self._data_path = str(data_path)
            probe           = np.load(self._data_path, mmap_mode="r")
            self._data_len  = probe.shape[0]
        else:
            self._data_path = str(self.temp_dir / "input_data.npy")
            if not Path(self._data_path).exists():
                logger.info("Saving input (%s, float32) to %s ...", data.shape, self._data_path)
                np.save(self._data_path, data.astype(DTYPE))
            self._data_len = data.shape[0]

        _threshold = flush_threshold or self.n_workers * 4
        self._scalar_writer = ScalarWriter(self.temp_dir, flush_threshold=_threshold)
        self._object_writer = ObjectWriter(
            self.temp_dir,
            stride=object_stride,
            flush_threshold=object_flush_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Submit one future per window. Already-completed windows are skipped.
        All workers stay fully utilised until the very last window.
        """
        completed = self._find_completed_windows()
        remaining = [i for i in range(self._data_len) if i not in completed]

        logger.info(
            "Windows total=%d  done=%d  remaining=%d  workers=%d",
            self._data_len, len(completed), len(remaining), self.n_workers,
        )

        if not remaining:
            logger.info("All windows already computed.")
            return

        self._scalar_writer.start()
        self._object_writer.start()

        try:
            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_worker_init,
                initargs=(self._calc_fn,),
            ) as pool:
                with tqdm(total=self._data_len, initial=len(completed),
                          desc="Windows", unit="win", position=0) as pbar:

                    pending   = {}          # future -> window_id
                    todo      = iter(remaining)
                    exhausted = False

                    def _fill():
                        nonlocal exhausted
                        while not exhausted and len(pending) < self.max_in_flight:
                            try:
                                window_id = next(todo)
                            except StopIteration:
                                exhausted = True
                                break
                            future = pool.submit(_worker, window_id, self._data_path)
                            pending[future] = window_id

                    _fill()

                    while pending:
                        done = [f for f in list(pending) if f.done()]
                        if not done:
                            time.sleep(0.005)
                            continue

                        for future in done:
                            pending.pop(future)
                            try:
                                window_id_out, obj, scalar_a, scalar_b = future.result()
                            except Exception:
                                logger.error("Worker failed", exc_info=True)
                                raise
                            self._scalar_writer.enqueue(window_id_out, scalar_a, scalar_b)
                            self._object_writer.enqueue(window_id_out, obj)
                            pbar.update(1)

                        _fill()

        finally:
            self._scalar_writer.stop(wait=True)
            self._object_writer.stop(wait=True)

    def collect_scalars(self) -> dict:
        """
        Merge all scalar checkpoint files.

        Returns
        -------
        dict mapping window_id -> (scalar_a, scalar_b), sorted by window_id.
        Raises RuntimeError if any window is missing.
        """
        all_files = sorted(self.temp_dir.glob("scalars_*.npy"))
        if not all_files:
            raise RuntimeError("No scalar checkpoints found. Did run() complete?")

        raw = {}
        for path in all_files:
            for row in np.load(path):
                raw[int(row["window_id"])] = (float(row["scalar_a"]), float(row["scalar_b"]))

        # use _data_len — not n_chunks * chunk_size — to avoid off-by-one
        # on the last partial chunk
        missing = set(range(self._data_len)) - raw.keys()
        if missing:
            raise RuntimeError(
                f"{len(missing)} window(s) missing scalars: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

        return dict(sorted(raw.items()))

    def collect_objects(self) -> dict:
        """
        Load all saved object checkpoints.

        Returns
        -------
        dict mapping window_id -> obj for every stride-boundary window.
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

    def _find_completed_windows(self) -> set:
        """Scan all scalar checkpoint files and return the set of done window_ids."""
        completed = set()
        for path in self.temp_dir.glob("scalars_*.npy"):
            try:
                for row in np.load(path):
                    completed.add(int(row["window_id"]))
            except Exception:
                pass
        return completed