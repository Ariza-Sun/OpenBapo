"""One inference child. No Discord token or audio is included in its result."""

import argparse
import ctypes
import json
import os
import signal
from pathlib import Path
from typing import Any

from openbapo.errors import InputError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    # Linux deployment: kill this child if its service parent disappears abruptly.
    # It inherits the instance lock FD as well, preventing a competing GPU worker.
    if os.name == "posix" and Path("/proc").exists():
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        expected_parent = os.environ.get("OPENBAPO_PARENT_PID")
        if expected_parent and os.getppid() != int(expected_parent):
            return 1
    request = json.loads((args.workdir / "request.json").read_text(encoding="utf-8"))
    payload: dict[str, Any]
    try:
        from openbapo.analysis.beats import BeatThisTracker
        from openbapo.analysis.pipeline import analyze_file
        from openbapo.analysis.separate import DemucsSeparator
        from openbapo.cli import DEFAULT_CHORD_EXPERTS, build_chord_stage, build_lyrics_stage

        result = analyze_file(
            args.workdir / request["audio"],
            separator=DemucsSeparator(device=args.device),
            tracker=BeatThisTracker(device=args.device),
            chord_stage=build_chord_stage(DEFAULT_CHORD_EXPERTS, args.device),
            lyrics_stage=build_lyrics_stage(args.device),
            lyrics_text=request.get("lyrics"),
            title=request["title"],
            artist=request.get("artist"),
        )
        payload = {
            "analysis": result.analysis.model_dump(mode="json"),
            "timings": result.timings.as_dict(),
        }
    except InputError as exc:
        payload = {"error": "long" if "Please trim" in str(exc) else "corrupt"}
    except Exception:
        # No exception repr: upstream exceptions can contain input text or tokens.
        payload = {"error": "service"}
    temporary = args.workdir / "result.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.workdir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
