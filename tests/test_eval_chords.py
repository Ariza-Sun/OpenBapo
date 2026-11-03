"""The chord evaluation script's loaders and scoring glue, without any model."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import write_tone

SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_chords.py"
spec = importlib.util.spec_from_file_location("eval_chords", SCRIPT)
assert spec and spec.loader
eval_chords = importlib.util.module_from_spec(spec)
sys.modules["eval_chords"] = eval_chords
spec.loader.exec_module(eval_chords)

mir_eval_missing = importlib.util.find_spec("mir_eval") is None


def jams_file(
    path: Path, *, chords: list[tuple[float, float, str]], key: str, beats: list[float]
) -> None:
    def ann(namespace: str, data: list[dict]) -> dict:
        return {
            "namespace": namespace,
            "annotation_metadata": {"data_source": "test"},
            "data": data,
        }

    chord_obs = [{"time": t, "duration": d, "value": v, "confidence": None} for t, d, v in chords]
    beat_obs = [
        {
            "time": t,
            "duration": 0.0,
            "value": {"position": i % 4 + 1, "measure": i // 4},
            "confidence": None,
        }
        for i, t in enumerate(beats)
    ]
    jam = {
        "annotations": [
            ann("chord", chord_obs),
            ann("chord", chord_obs[:1]),  # the "performed" version
            ann("key_mode", [{"time": 0.0, "duration": 30.0, "value": key, "confidence": None}]),
            ann("beat_position", beat_obs),
        ],
        "file_metadata": {"duration": 30.0},
    }
    path.write_text(json.dumps(jam), encoding="utf-8")


def test_guitarset_loader(tmp_path: Path) -> None:
    root = tmp_path / "guitarset"
    (root / "annotation").mkdir(parents=True)
    (root / "audio_mono-mic").mkdir()
    for name in ("00_BN1-129-Eb_comp", "03_Rock2-142-D_solo"):
        jams_file(
            root / "annotation" / f"{name}.jams",
            chords=[(0.0, 2.0, "C:maj"), (2.0, 2.0, "G:maj/3")],
            key="C:major",
            beats=[0.0, 0.5, 1.0, 1.5, 2.0],
        )
        write_tone(root / "audio_mono-mic" / f"{name}_mic.wav", seconds=0.2)
    tracks = eval_chords.guitarset_tracks(root)
    assert [t.name for t in tracks] == ["00_BN1-129-Eb_comp", "03_Rock2-142-D_solo"]
    assert [t.split for t in tracks] == ["tune", "report"]
    assert [t.group for t in tracks] == ["BN", "Ro"]
    first = tracks[0]
    assert first.chord_labels == ["C:maj", "G:maj/3"]
    np.testing.assert_allclose(first.chord_intervals, [[0.0, 2.0], [2.0, 4.0]])
    assert first.key == (0, "major")
    assert first.beats.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0] and first.downbeats.tolist() == [
        0.0,
        2.0,
    ]
    assert first.performed_labels == ["C:maj"]  # the second chord annotation rides along
    np.testing.assert_allclose(first.performed_intervals, [[0.0, 2.0]])
    performed = eval_chords.guitarset_tracks(root, chords="performed")
    assert performed[0].chord_labels == ["C:maj"] and performed[0].performed_labels is None


def test_jams_dict_of_lists_form() -> None:
    ann = {
        "data": {
            "time": [0.0, 1.0],
            "duration": [1.0, 1.0],
            "value": ["C:maj", "N"],
            "confidence": [None, None],
        }
    }
    obs = eval_chords.jams_observations(ann)
    assert obs[1] == {"time": 1.0, "duration": 1.0, "value": "N", "confidence": None}


AAM_BEATINFO = (
    "@RELATION 'Beat information of 0001'\n"
    "@ATTRIBUTE 'Start time in seconds' NUMERIC\n@ATTRIBUTE 'Bar count' NUMERIC\n"
    "@ATTRIBUTE 'Quarter count' NUMERIC\n@ATTRIBUTE 'Chord name' STRING\n@DATA\n"
    "0.0,1,1,'Fmaj'\n0.5,1,2,'Fmaj'\n2.0,2,1,'A#maj'\n4.0,3,1,'Amin'\n5.0,3,3,'Dmin'\n8.0,5,1,'Cmaj'\n"
)
AAM_SEGMENTS = (
    "@RELATION 'Segment information of 0001'\n"
    "@ATTRIBUTE 'Start time in seconds' NUMERIC\n@ATTRIBUTE 'Mark' STRING\n"
    "@ATTRIBUTE 'Tempo' NUMERIC\n@ATTRIBUTE 'Key' STRING\n@ATTRIBUTE 'Instruments' STRING\n"
    "@ATTRIBUTE 'Generator' STRING\n@DATA\n"
    "0.0,'A',120,'Fmaj','[TenorSax,Ukulele]','[MelodyBow,ChordArpeggios]'\n"
    "6.0,'B',120,'Gmin','[Flugelhorn,Drums]','[MelodyBow,BassLine]'\n"
    "10.0,'end',0,'','[]','[]'\n"
)


def write_aam(root: Path, track_id: str, audio_name: str | None, annotations: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if annotations:
        beatinfo = AAM_BEATINFO.replace("0001", track_id)
        (root / f"{track_id}_beatinfo.arff").write_text(beatinfo, encoding="utf-8")
        segments = AAM_SEGMENTS.replace("0001", track_id)
        (root / f"{track_id}_segments.arff").write_text(segments, encoding="utf-8")
    (root / f"{track_id}_onsets.arff").write_text(
        "@RELATION onsets\n@ATTRIBUTE 'Onset time in seconds' NUMERIC\n@DATA\n0.0\n",
        encoding="utf-8",
    )
    if audio_name:
        write_tone(root / audio_name, seconds=0.2)


def test_aam_loader_reads_the_v110_layout(tmp_path: Path) -> None:
    root = tmp_path / "aam"
    write_aam(root / "annotations", "0001", None)
    (root / "mixes").mkdir()
    write_tone(root / "mixes" / "0001_mix.wav", seconds=0.2)
    write_aam(root / "annotations", "0002", None, annotations=False)  # onsets only
    write_aam(root / "annotations", "0003", None)  # no audio
    write_aam(root / "annotations", "0004", "0004.wav")  # audio without "mix" in the name
    tracks = eval_chords.aam_tracks(root, count=500)
    assert [t.name for t in tracks] == ["0001", "0004"]
    t = tracks[0]
    assert t.audio == root / "mixes" / "0001_mix.wav"
    assert t.chord_labels == ["F:maj", "A#:maj", "A:min", "D:min", "C:maj"]
    np.testing.assert_allclose(t.chord_intervals, [[0, 2], [2, 4], [4, 5], [5, 8], [8, 10]])
    assert t.key == (5, "major")  # F major lasts 6 s, G minor 4 s
    assert t.key_share == pytest.approx(0.6)
    assert t.key_times == {"F major": pytest.approx(6.0), "G minor": pytest.approx(4.0)}
    assert eval_chords.key_hit_major(t, (7, "minor")) == 1.0  # G minor holds 40% of the piece
    assert eval_chords.key_hit_major(t, (0, "major")) == 0.0
    assert eval_chords.key_hit_major(t, None) == 0.0
    np.testing.assert_allclose(t.beats, np.arange(0, 10, 0.5))
    np.testing.assert_allclose(t.downbeats, [0, 2, 4, 6, 8])
    assert eval_chords.aam_tracks(root, count=1)[0].name == "0001"
    assert eval_chords.aam_tracks(root, count=0) == []


def test_arff_reader_accepts_a_missing_data_line(tmp_path: Path) -> None:
    path = tmp_path / "x.arff"
    path.write_text(
        "@RELATION 'Beat information of 0001'\n@ATTRIBUTE 'Start time in seconds' NUMERIC\n"
        "@ATTRIBUTE 'Chord name' STRING\n\n0.0,'Fmaj'\n% comment\n0.5,'Gmin'\n",
        encoding="utf-8",
    )
    assert eval_chords.read_arff(path) == (
        ["Start time in seconds", "Chord name"],
        [["0.0", "Fmaj"], ["0.5", "Gmin"]],
    )
    path.write_text("@ATTRIBUTE a NUMERIC\n@DATA\n@1\n", encoding="utf-8")
    assert eval_chords.read_arff(path) == (["a"], [["@1"]])


def test_inspect_reports_unparsable_labels_in_any_track(capsys: pytest.CaptureFixture[str]) -> None:
    def track(name: str, labels: list[str]) -> eval_chords.Track:
        n = len(labels)
        return eval_chords.Track(
            dataset="aam", split="report", name=name, group="aam", audio=Path("x.wav"),
            chord_intervals=np.array([[i, i + 1] for i in range(n)], dtype=float),
            chord_labels=labels, key=None, beats=np.zeros(0), downbeats=np.zeros(0),
        )  # fmt: skip

    eval_chords.inspect([track("0001", ["C:maj", "N"]), track("0007", ["N.C.", "G:min"])])
    out = capsys.readouterr().out
    assert "[aam] 0001" in out and "WARNING unparsable label 'N.C.' in 1 tracks, e.g. 0007" in out


def test_aam_meter_and_key_fallbacks() -> None:
    anchors = [(0.0, 1, 1, "C"), (1.5, 2, 1, "F"), (3.0, 3, 1, "G"), (4.0, 3, 3, "C")]
    sections = [(0.0, 120.0, "Cmaj")]
    assert eval_chords.aam_beats_per_bar(anchors, sections) == 3
    assert eval_chords.aam_beats_per_bar([(0.0, 1, 1, "C"), (1.0, 1, 2, "F")], sections) == 2
    beats, downbeats = eval_chords.aam_beat_grid(sections, 4.5, anchors)
    np.testing.assert_allclose(beats, np.arange(0, 4.5, 0.5))
    np.testing.assert_allclose(downbeats, [0, 1.5, 3.0])
    assert eval_chords.aam_key([(0.0, 120.0, ""), (2.0, 120.0, "Amin")], 4.0) == ((9, "minor"), 1.0)
    assert eval_chords.aam_key([(0.0, 120.0, "")], 4.0) == (None, 1.0)
    key, share = eval_chords.aam_key([(0.0, 120.0, "Fmaj"), (6.0, 120.0, "Gmin")], 10.0)
    assert key == (5, "major") and share == pytest.approx(0.6)
    intervals, labels = eval_chords.aam_chord_segments([(0.0, 1, 1, "N"), (2.0, 2, 1, "G7")], 2.0)
    assert labels == ["N"] and intervals.tolist() == [[0.0, 2.0]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A:minor", (9, "minor")),
        ("Eb:major", (3, "major")),
        ("F# minor", (6, "minor")),
        ("Cm", (0, "minor")),
        ("G", (7, "major")),
        ("N", None),
        ("??", None),
    ],
)
def test_parse_key(text: str, expected: tuple[int, str] | None) -> None:
    assert eval_chords.parse_key(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C", "C:maj"),
        ("Am", "A:min"),
        ("G7", "G:7"),
        ("Bbm7", "Bb:min7"),
        ("F#dim", "F#:dim"),
        ("N", "N"),
        ("N.C.", "N"),
        ("n.c", "N"),
        ("D:maj/3", "D:maj/3"),
        ("Cm7b5", "C:hdim7"),
    ],
)
def test_normalize_aam_label(text: str, expected: str) -> None:
    assert eval_chords.normalize_aam_label(text) == expected


@pytest.mark.skipif(mir_eval_missing, reason="mir_eval not installed")
def test_chord_scores_self_check_and_gaps() -> None:
    ref = np.array([[0.0, 2.0], [2.0, 4.0]])
    labels = ["C:maj", "G:maj/3"]
    perfect = eval_chords.chord_scores(ref, labels, ref, labels)
    assert all(perfect[m] == 1.0 for m in eval_chords.CHORD_METRICS)
    # An estimate covering only the first half: majmin credit for half the time.
    half = eval_chords.chord_scores(ref, labels, np.array([[0.0, 2.0]]), ["C:maj"])
    assert half["majmin"] == pytest.approx(0.5)
    # Ignoring the inversion costs the _inv metrics but not majmin.
    no_inv = eval_chords.chord_scores(ref, labels, ref, ["C:maj", "G:maj"])
    assert no_inv["majmin"] == 1.0 and no_inv["majmin_inv"] == pytest.approx(0.5)
    assert np.isnan(eval_chords.chord_scores(np.zeros((0, 2)), [], ref, labels)["majmin"])


@pytest.mark.skipif(mir_eval_missing, reason="mir_eval not installed")
def test_key_scores() -> None:
    assert eval_chords.key_scores((0, "major"), (0, "major")) == {
        "key_exact": 1.0,
        "key_weighted": 1.0,
        "key_ref": "C major",
        "key_est": "C major",
        "key_relation": "exact",
    }
    relative = eval_chords.key_scores((0, "major"), (9, "minor"))
    assert relative["key_exact"] == 0.0 and 0.0 < relative["key_weighted"] < 1.0
    assert relative["key_relation"] == "relative"
    assert relative["key_ref"] == "C major" and relative["key_est"] == "A minor"
    assert eval_chords.key_scores((0, "major"), (7, "major"))["key_relation"] == "fifth"
    assert eval_chords.key_scores((0, "major"), (0, "minor"))["key_relation"] == "parallel"
    assert eval_chords.key_scores((0, "major"), (1, "major"))["key_relation"] == "other"
    assert eval_chords.key_scores((0, "major"), (0, "major"))["key_relation"] == "exact"
    none = eval_chords.key_scores((0, "major"), None)
    assert (
        none["key_exact"] == 0.0 and none["key_weighted"] == 0.0 and none["key_relation"] == "none"
    )
    assert np.isnan(eval_chords.key_scores(None, (0, "major"))["key_exact"])


def test_summarize_splits_by_dataset_and_group() -> None:
    rows = [
        {
            "track": "a",
            "dataset": "guitarset",
            "split": "tune",
            "group": "BN",
            "majmin": 0.8,
            "sevenths": 0.7,
            "majmin_inv": 0.8,
            "sevenths_inv": 0.7,
            "root": 0.9,
            "key_exact": 1.0,
            "key_weighted": 1.0,
            "seconds": 1.0,
        },
        {
            "track": "b",
            "dataset": "guitarset",
            "split": "report",
            "group": "Ro",
            "majmin": 0.6,
            "sevenths": 0.5,
            "majmin_inv": 0.6,
            "sevenths_inv": 0.5,
            "root": 0.7,
            "key_exact": 0.0,
            "key_weighted": 0.5,
            "seconds": 2.0,
        },
        {
            "track": "c",
            "dataset": "aam",
            "split": "report",
            "group": "aam",
            "majmin": 0.9,
            "sevenths": 0.9,
            "majmin_inv": 0.9,
            "sevenths_inv": 0.9,
            "root": 0.9,
            "key_exact": 1.0,
            "key_weighted": 1.0,
            "seconds": 3.0,
        },
    ]
    rows[0]["track"] = "00_BN1-129-Eb_comp"
    rows[0]["majmin_performed"], rows[0]["root_performed"] = 0.9, 1.0
    rows[1]["track"] = "03_Rock1-130-A_solo"
    rows[1]["key_relation"] = "relative"
    rows[2]["key_relation"], rows[2]["key_share"] = "other", 0.6  # a modulating piece
    rows[2]["key_exact"], rows[2]["key_hit_major"] = 0.0, 1.0  # ...that hit its second key
    rows.append(dict(rows[2], track="d", key_relation="exact", key_share=1.0, key_exact=1.0))
    summary = eval_chords.summarize({"rows": rows})
    assert summary["guitarset_report"]["key_relations"]["relative"] == 1
    assert summary["aam"]["key_relations"] == {
        "exact": 1, "fifth": 0, "relative": 0, "parallel": 0, "other": 1, "none": 0
    }  # fmt: skip
    assert (
        summary["aam"]["key_exact_single_key"] == 1.0 and summary["aam"]["single_key_tracks"] == 1
    )
    assert "key_exact_single_key" not in summary["guitarset_report"]
    assert summary["guitarset_tune_comp"]["majmin_performed"] == 0.9
    assert "majmin_performed" not in summary["aam"]  # no performed annotation there
    assert summary["aam"]["key_hit_major_modulating"] == 1.0
    assert summary["aam"]["modulating_tracks"] == 1 and summary["aam"]["key_exact"] == 0.5
    assert summary["guitarset_tune"]["majmin"] == 0.8 and summary["guitarset_tune"]["tracks"] == 1
    assert summary["guitarset_tune_comp"]["tracks"] == 1 and "guitarset_tune_solo" not in summary
    assert summary["guitarset_report_solo"]["majmin"] == 0.6
    assert "guitarset_report_comp" not in summary
    assert summary["guitarset_report"]["key_exact"] == 0.0
    assert summary["aam"]["seconds"] == 3.0 and summary["aam"]["tracks"] == 2
    text = eval_chords.markdown(
        [{"system": "committee (tuned)", "summary": summary, "params": {}, "rows": rows}],
        {"date": "2026-01-01", "python": "3", "device": "cpu", "packages": {}},
        ["note"],
    )
    assert "| committee (tuned) | 0.600 |" in text and "- note" in text
    assert "| aam | 1 | 0 | 0 | 0 | 1 | 0 | 1.000" in text and text.count("1 首") >= 2
    assert "转调曲目命中任一主要调" in text
    assert "对实奏和弦" in text and "| committee (tuned) | 0.900 | 1.000 |" in text
    assert "| 2.00 s | 参考 |" in text  # the solo-only and mixed GuitarSet rows are reference
    assert "| 3.00 s | 通过 |" in text  # gated on the single-key piece although key_exact is 0.5
    solo_row = (
        "| committee (tuned) | 0.600 | 0.500 | 0.600 | 0.500 | 0.700 | 0.000 | 0.500 | 2.00 s"
    )
    assert "仅独奏" in text and solo_row + " | 参考 |" in text


def test_sheet_loader_reads_renders(tmp_path: Path) -> None:
    from opencapo.sheet import parse_sheet, render, write_reference, write_wav

    sheet = parse_sheet("title: t\nkey: D\ncapo: 2\nbpm: 120\n---\nC | 1 2 3 4\nG | 5 - - -\n")
    rendered = render(sheet, melody=False)
    write_wav(tmp_path / "t.wav", rendered.samples, rendered.sample_rate)
    write_reference(tmp_path / "t.json", rendered.reference)
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")  # not a reference
    tracks = eval_chords.sheet_tracks(tmp_path)
    assert [t.name for t in tracks] == ["t"]
    t = tracks[0]
    assert t.dataset == "sheets" and t.group == "t" and t.key == (2, "major")
    assert t.chord_labels == ["D:maj", "A:maj"]
    np.testing.assert_allclose(t.chord_intervals, [[0.0, 2.0], [2.0, 4.0]])
    assert len(t.beats) == 8 and t.downbeats.tolist() == [0.0, 2.0]
    assert t.key_times == {"D major": 4.0} and t.key_share == 1.0
