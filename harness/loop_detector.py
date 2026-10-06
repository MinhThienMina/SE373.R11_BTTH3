"""Loop/stall detection helpers.

This follows the lecture slide idea: detect repeated actions by comparing
``(tool, args)`` and detect stalls by watching a domain-specific progress value.
"""

import json
from collections import Counter
from collections import deque


class LoopDetector:
    """Detect repeated actions and lack of progress in a small recent window."""

    def __init__(self, window=6, repeat_k=3, stall_n=5):
        self.recent = deque(maxlen=window)
        self.repeat_k = repeat_k
        self.stall_n = stall_n
        self.last_progress = None
        self.stall = 0

    def check(self, tool, args, progress):
        """Return ``LOOP_DETECTED``, ``STALL_DETECTED``, or ``None``.

        ``progress`` is supplied by the application, for example number of valid
        candidates found, whether a hold exists, or whether a booking exists.
        There is no universal progress metric for every agent.
        """
        fp = (tool, json.dumps(args, sort_keys=True, default=str))
        if self.recent.count(fp) + 1 >= self.repeat_k:
            return "LOOP_DETECTED"
        self.recent.append(fp)

        if progress == self.last_progress:
            self.stall += 1
        else:
            self.stall = 0
        self.last_progress = progress

        if self.stall >= self.stall_n:
            return "STALL_DETECTED"
        return None


def repeated_call_count(audit):
    """Return counts keyed by (tool, args-json) from a guard audit list."""
    counts = Counter()
    for item in audit:
        key = (item.get("tool"), json.dumps(item.get("args", {}), sort_keys=True, default=str))
        counts[key] += 1
    return counts


__all__ = ["LoopDetector", "repeated_call_count"]
