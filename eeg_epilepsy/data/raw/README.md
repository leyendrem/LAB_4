# Datos originales

El EDF de 180 MB no se distribuye dentro del ZIP. Descárgalo de forma reproducible desde la raíz del proyecto con:

```powershell
uv sync
uv run python scripts/download_data.py
```

Se descargan `chb01_03.edf` y `chb01-summary.txt` desde la base CHB-MIT de PhysioNet. `download_manifest.json` registra URL, tamaño y SHA-256. El archivo EDF original se conserva completo; las exclusiones de canales solo afectan al conjunto de canales analizados.
