"""Allow CPU-only unittest discovery directly from a source checkout."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def requires_local_artifacts(*paths):
    """Private experiment integration tests remain runnable in the author's workspace."""
    return unittest.skipUnless(all(Path(path).exists() for path in paths),
                               "Requires local experiment artifacts; not included in the public repository")
