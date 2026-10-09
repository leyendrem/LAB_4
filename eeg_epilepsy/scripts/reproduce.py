from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Permite importar el paquete ubicado en src/.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_epilepsy.data.download import download_data
from eeg_epilepsy.data.audit import main as audit_data
from eeg_epilepsy.data.check_channels import main as check_channels


def main():
    print("\n1. DESCARGA")
    download_data()

    print("\n2. AUDITORÍA")
    audit_data()

    print("\n3. COMPARACIÓN DE CANALES Y TRAZADO CRUDO")
    check_channels()

    print("\nPROCESO FINALIZADO")


if __name__ == "__main__":
    main()