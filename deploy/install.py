"""Install the tested inference stack without lv-chordia replacing CUDA torch.

lv-chordia declares torch>=2.13, but the existing model regression runs all five
networks on 2.5.1. Only that package uses --no-deps; install its other dependencies
explicitly. This compatibility exception is documented in self-hosting.md.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "pyproject.toml").read_text())
requirements = [
    r
    for r in project["project"]["optional-dependencies"]["analysis"]
    if not r.startswith("lv-chordia")
]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-c",
        str(root / "deploy/constraints.txt"),
        *requirements,
        "pretty-midi>=0.2.9",
        "pydub>=0.23.1",
        "-e",
        f"{root}[bot,render]",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "lv-chordia @ git+https://github.com/openmirlab/lv-chordia.git@d8ab9ac99bce081bb79065d717f96e2c3b2099c8",
    ],
    check=True,
)
