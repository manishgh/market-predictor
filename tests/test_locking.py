"""R3: portable file locking."""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.locking import LockTimeout, file_lock


class FileLockTest(unittest.TestCase):
    def test_process_death_releases_lock_without_manual_cleanup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "artifact.bin"
            ready = root / "ready"
            script = (
                "import time\n"
                "from pathlib import Path\n"
                "from market_predictor.locking import file_lock\n"
                f"target = Path({str(target)!r})\n"
                f"ready = Path({str(ready)!r})\n"
                "with file_lock(target):\n"
                "    ready.write_text('ready', encoding='utf-8')\n"
                "    time.sleep(60)\n"
            )
            process = subprocess.Popen([sys.executable, "-c", script])
            try:
                deadline = time.monotonic() + 5.0
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), "child process did not acquire lock")
                process.kill()
                process.wait(timeout=5)
                with file_lock(target, timeout=1.0):
                    pass
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_lock_is_exclusive_then_reacquirable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "artifact.bin"
            with file_lock(target):
                with self.assertRaises(LockTimeout):
                    with file_lock(target, timeout=0.1):
                        pass
            # Once released the lock can be re-acquired.
            with file_lock(target, timeout=0.5):
                pass


if __name__ == "__main__":
    unittest.main()
