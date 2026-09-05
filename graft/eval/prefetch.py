"""Single-thread prefetch wrapper to overlap per-frame I/O with GPU work.

The worker thread eagerly pulls the next item(s) from the underlying iterator
while the consumer is still busy on the current item. Useful when the iterator
does CPU/IO-bound work (Image.open, trimesh.load, XML parse) — that work runs
in parallel with the consumer's GPU work.

Caveat: if the iterator itself launches GPU kernels (e.g. SAM inside
iter_rich_frames), CUDA serialises across threads on the same context, so
overlap with the consumer's GPU work is limited. Still useful for the CPU
parts that surround the kernel launches.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def prefetch_iter(it: Iterable[T], max_ahead: int = 2) -> Iterator[T]:
    sentinel = object()
    q: queue.Queue = queue.Queue(maxsize=max_ahead)

    def _worker():
        try:
            for x in it:
                q.put(x)
        except Exception as e:  # propagate to consumer
            q.put(("__err__", e))
            return
        q.put(sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        x = q.get()
        if x is sentinel:
            break
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "__err__":
            raise x[1]
        yield x
    t.join()
