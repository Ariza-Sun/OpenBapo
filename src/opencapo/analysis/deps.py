"""Dependency check for the analysis extra. Must stay importable without numpy or torch."""

import importlib.util

from opencapo.errors import InputError

# Importable names of the analysis extra, checked before any model is loaded
# so a missing install produces one clear message instead of a traceback.
ANALYSIS_PACKAGES = ("numpy", "librosa", "torch", "demucs", "beat_this", "lv_chordia")


def missing_analysis_packages() -> list[str]:
    return [name for name in ANALYSIS_PACKAGES if importlib.util.find_spec(name) is None]


def require_analysis_packages() -> None:
    missing = missing_analysis_packages()
    if missing:
        raise InputError(
            f"analysis needs {', '.join(missing)}; install them with "
            'pip install "opencapo[analysis]" (and the ffmpeg command-line tools)'
        )
