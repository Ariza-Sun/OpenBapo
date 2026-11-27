#!/bin/sh
set -eu
umask 077
# CTranslate2 needs the CUDA libraries supplied by the torch wheel at runtime.
cuda_libs=$(python -c 'import site; from pathlib import Path; print(":".join(str(p) for root in site.getsitepackages() for p in Path(root).glob("nvidia/*/lib")))')
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
