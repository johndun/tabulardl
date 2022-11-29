import time


class Timer:
    """Callable that maintains a list of timestamps.

    Attributes:
        times: A list of times the class instance was called, including
            initialization.
        total_time: Total accumulated time.
    """
    def __init__(self):
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, include_in_total: bool = True):
        """Appends a new time and returns most recent delta."""
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t
