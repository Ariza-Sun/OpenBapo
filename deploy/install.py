"""Install the tested inference stack at the versions in deploy/constraints.txt.

Install the CUDA build of torch first. lv-chordia, published only on GitHub, is
installed pinned to a commit without its dependencies, so it cannot swap that
torch build for another; its other dependencies are installed explicitly.
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
