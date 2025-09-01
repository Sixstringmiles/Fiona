# ==============================
# File: src/utils.py
# ==============================
import logging
import queue

from .config import CFG


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, CFG.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RingBuffer:
    """Simple thread-safe ring-ish buffer for amplitude/viseme data."""
    def __init__(self, maxlen=256):
        self.q = queue.Queue(maxsize=maxlen)

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            # Drop oldest by getting one item
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def try_pop(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

