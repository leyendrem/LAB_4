"""Descarga reproducible del EEG CHB-MIT y su resumen oficial."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_epilepsy.data.download import download_data

if __name__ == "__main__":
    download_data()
