"""Download all inference weights before accepting Discord uploads."""

import argparse
import gc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    from openbapo.analysis.beats import BeatThisTracker
    from openbapo.analysis.chords.models.btc import BTCExpert
    from openbapo.analysis.chords.models.lvchordia import LVChordiaExpert
    from openbapo.analysis.lyrics.align import wav2vec2_aligner
    from openbapo.analysis.lyrics.recognize import load_whisper
    from openbapo.analysis.separate import DemucsSeparator

    # These are the existing adapters' lazy loaders, not a second set of model URLs.
    for adapter in (DemucsSeparator, BeatThisTracker, BTCExpert, LVChordiaExpert):
        print(f"Loading {adapter.__name__}", flush=True)
        instance = adapter(device=args.device)
        instance._load()
        del instance
        gc.collect()
    print("Loading Whisper", flush=True)
    whisper = load_whisper(device=args.device)
    del whisper
    gc.collect()
    for language in ("en", "zh"):
        print(f"Loading alignment: {language}", flush=True)
        aligner = wav2vec2_aligner(args.device)
        aligner.model(language)
        del aligner
        gc.collect()
    print("Model cache ready", flush=True)


if __name__ == "__main__":
    main()
