# Externals

Tato složka obsahuje externí závislosti které nejsou součástí pip.

## LPM SDK
Zkopíruj nebo symlinkuj LPM SDK do `externals/LPM/`:
- `externals/LPM/lib/x64/lpm-v7.dll`
- `externals/LPM/modules-v7/`

Cesta se konfiguruje v `configs/config.yaml` pod klíčem `lpm.sdk_root`.
Pokud není nastavena, aplikace hledá SDK automaticky relativně od umístění skriptu.
