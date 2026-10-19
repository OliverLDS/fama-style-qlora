"""CPU-safe entry point for the original Gemma 3 1B training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fama_style.training.train import main


if __name__ == "__main__":
    main()
