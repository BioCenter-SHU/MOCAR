"""Runtime helpers for timestamped logs and test-only checkpoint loading."""

import datetime
from pathlib import Path
import sys

_OPEN_LOGS = []

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

def configure_run(args, task_name):
    if getattr(args, 'checkpoint', None):
        args.eval_only = True

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = 'eval' if getattr(args, 'eval_only', False) else 'train'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / '{}_{}_{}.log'.format(timestamp, task_name, mode)
    log_handle = log_path.open('w', encoding='utf-8', buffering=1)
    _OPEN_LOGS.append(log_handle)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    print('Log file: {}'.format(log_path))
    return log_path
