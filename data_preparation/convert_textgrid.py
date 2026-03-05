#!/usr/bin/env python3
"""Convert PriMock57 TextGrid transcripts to readable text and structured JSON.

Reads paired doctor/patient TextGrid files, merges utterances chronologically,
strips markup tags, and outputs:
  - Readable transcript: ``[MM:SS] Speaker: text`` per line
  - Structured JSON: one file per consultation (``{id}.json``) with call_id, transcript, etc.

Requires the ``textgrid`` package (install via ``uv sync --extra data-prep``).

Example
-------
Convert all consultations in a directory::

    python convert_textgrid.py --input-dir data/textgrids --output-dir output/

Convert specific consultations::

    python convert_textgrid.py --input-dir data/textgrids --output-dir output/ \\
        --consultations day1_consultation02 day1_consultation04
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import textgrid


_TEXTGRID_TAGS = ["<UNSURE>", "</UNSURE>", "<UNIN/>", "<INAUDIBLE_SPEECH/>"]


def _get_utterances(tg_path: str) -> list[dict]:
    """Extract non-empty utterances from a TextGrid file."""
    tg = textgrid.TextGrid()
    tg.read(tg_path)
    utterances = []
    for tier in tg.tiers:
        for interval in tier.intervals:
            if len(interval.mark) > 0:
                utterances.append(
                    {"text": interval.mark, "from": interval.minTime, "to": interval.maxTime}
                )
    return utterances


def _strip_tags(text: str) -> str:
    """Remove TextGrid markup tags and collapse whitespace."""
    for tag in _TEXTGRID_TAGS:
        text = text.replace(tag, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to ``[MM:SS]`` format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def get_timestamped_transcript(
    doctor_path: str, patient_path: str
) -> tuple[str, list[dict]]:
    """Merge doctor/patient TextGrid utterances into a timestamped transcript.

    Consecutive utterances from the same speaker are merged into a single turn.

    Returns
    -------
    transcript : str
        Readable transcript with ``[MM:SS] Speaker: text`` lines.
    merged : list[dict]
        List of merged utterance dicts (speaker, from, to, text).
    """
    utterances_doctor = _get_utterances(doctor_path)
    utterances_patient = _get_utterances(patient_path)

    for u in utterances_doctor:
        u["speaker"] = "Doctor"
    for u in utterances_patient:
        u["speaker"] = "Patient"

    combined = sorted(utterances_doctor + utterances_patient, key=lambda x: x["from"])

    merged: list[dict] = []
    current: dict | None = None

    for u in combined:
        text = _strip_tags(u["text"])
        if not text:
            continue

        if current is None:
            current = {"speaker": u["speaker"], "from": u["from"], "to": u["to"], "text": text}
        elif current["speaker"] == u["speaker"]:
            current["text"] += " " + text
            current["to"] = u["to"]
        else:
            merged.append(current)
            current = {"speaker": u["speaker"], "from": u["from"], "to": u["to"], "text": text}

    if current is not None:
        merged.append(current)

    lines = [f"{_format_timestamp(u['from'])} {u['speaker']}: {u['text']}" for u in merged]
    return "\n".join(lines), merged


def convert_consultation(
    consultation_id: str, doctor_path: str, patient_path: str
) -> dict:
    """Build a structured JSON dict for a single consultation."""
    transcript, _merged = get_timestamped_transcript(doctor_path, patient_path)
    return {
        "call_id": consultation_id,
        "dataset": "primock57",
        "recording_url": f"https://github.com/babylonhealth/primock57/blob/main/audio/{consultation_id}_patient.wav",
        "transcript_golden": transcript,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert PriMock57 TextGrid transcripts to readable text + JSON."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_doctor.TextGrid and *_patient.TextGrid files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Per-consultation {id}_readable.txt and {id}.json "
        "files are written here.",
    )
    parser.add_argument(
        "--consultations",
        nargs="+",
        metavar="ID",
        help="Process only these consultation IDs (e.g. day1_consultation02). "
        "Default: process all consultations found in --input-dir.",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.consultations:
        doctor_paths = []
        for cid in args.consultations:
            p = os.path.join(args.input_dir, f"{cid}_doctor.TextGrid")
            if os.path.exists(p):
                doctor_paths.append(p)
            else:
                print(f"Warning: TextGrid not found for {cid}: {p}")
        print(f"Processing {len(doctor_paths)} specified consultation(s)")
    else:
        doctor_paths = sorted(glob(os.path.join(args.input_dir, "*_doctor.TextGrid")))
        print(f"Found {len(doctor_paths)} consultation(s)")

    count = 0
    for dp in doctor_paths:
        cid = Path(dp).name.replace("_doctor.TextGrid", "")
        pp = dp.replace("_doctor.TextGrid", "_patient.TextGrid")

        if not os.path.exists(pp):
            print(f"Warning: patient TextGrid not found for {cid}, skipping")
            continue

        print(f"Converting {cid}...")

        consultation_data = convert_consultation(cid, dp, pp)

        txt_path = os.path.join(args.output_dir, f"{cid}_readable.txt")
        with open(txt_path, "w") as f:
            f.write(consultation_data["transcript_golden"])
        print(f"  -> {txt_path}")

        json_path = os.path.join(args.output_dir, f"{cid}.json")
        with open(json_path, "w") as f:
            json.dump(consultation_data, f, indent=2)
        print(f"  -> {json_path}")

        count += 1

    print(f"Conversion complete — {count} consultation(s) processed.")


if __name__ == "__main__":
    main()
