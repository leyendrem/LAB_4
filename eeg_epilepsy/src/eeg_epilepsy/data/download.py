from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil
from datetime import datetime, timezone

# Localiza el proyecto independientemente de dónde ejecutes el comando.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

#BASE_URL = "https://physionet.org/content/chbmit/1.0.0/"
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/"
FILES = [
    "chb01_03.edf",
    "chb01-summary.txt",
]


def sha256_file(path):
    """Calcula una huella del contenido para registrar su integridad."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "download_manifest.json"

    # Conserva las fechas de las descargas anteriores.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    for filename in FILES:
        destination = DATA_DIR / filename
        url = BASE_URL + filename

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Ya existe: {filename}")
        else:
            print(f"Descargando: {filename}")

            # Un archivo incompleto no se confunde con una descarga terminada.
            temporary = destination.with_suffix(destination.suffix + ".part")

            try:
                with urlopen(url, timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)

                if temporary.stat().st_size == 0:
                    raise ValueError(f"La descarga está vacía: {filename}")

                temporary.replace(destination)

            finally:
                temporary.unlink(missing_ok=True)

            manifest[filename] = {
                "dataset": "CHB-MIT Scalp EEG Database",
                "version": "1.0.0",
                "url": url,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }

        current_hash = sha256_file(destination)
        previous = manifest.get(filename, {})
        previous_hash = previous.get("sha256")

        if previous_hash and previous_hash != current_hash:
            raise ValueError(
                f"El contenido de {filename} cambió respecto al manifiesto."
            )

        manifest[filename] = {
            **previous,
            "dataset": "CHB-MIT Scalp EEG Database",
            "version": "1.0.0",
            "url": url,
            # Si ya existía sin manifiesto, su fecha de descarga es desconocida.
            "downloaded_at_utc": previous.get("downloaded_at_utc"),
            "relative_path": destination.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": current_hash,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("Datos disponibles en data/raw/")


if __name__ == "__main__":
    download_data()