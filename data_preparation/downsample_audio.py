#!/usr/bin/env python3
"""Downsample WAV audio files from a source sample rate to 8 kHz.

Uses the ``soxr`` library for high-quality resampling.  Designed for
PriMock57 patient audio (16 kHz mono 16-bit WAV) but works with any WAV
file matching the expected format.

Requires ``soxr`` and ``numpy`` (install via ``uv sync --extra data-prep``).

Example
-------
Downsample all patient audio in a directory::

    python downsample_audio.py --input-dir audio/ --output-dir audio_8kHz/

Downsample specific files::

    python downsample_audio.py --files a.wav b.wav --output-dir out/
"""

from __future__ import annotations

import argparse
import os
import wave
from glob import glob
from pathlib import Path

import numpy as np
import soxr

TARGET_SAMPLE_RATE = 8000


def downsample(audio_bytes: bytes, input_rate: int) -> bytes:
    """Resample 16-bit PCM audio bytes from *input_rate* to 8 kHz."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    resampled = soxr.resample(samples, input_rate, TARGET_SAMPLE_RATE, quality="HQ")
    return resampled.astype(np.int16).tobytes()


def downsample_file(input_path: str, output_path: str, expected_rate: int = 16000) -> bool:
    """Downsample a single WAV file to 8 kHz.

    Validates that the input is mono, 16-bit, at *expected_rate*.
    Returns ``True`` on success, ``False`` if the file was skipped.
    """
    with wave.open(input_path, "rb") as wf:
        if wf.getframerate() != expected_rate:
            print(f"  Skipped: sample rate is {wf.getframerate()} Hz, expected {expected_rate} Hz")
            return False
        if wf.getnchannels() != 1:
            print(f"  Skipped: {wf.getnchannels()} channels, expected mono")
            return False
        if wf.getsampwidth() != 2:
            print(f"  Skipped: {wf.getsampwidth()}-byte samples, expected 2 (16-bit)")
            return False
        audio_data = wf.readframes(wf.getnframes())

    resampled = downsample(audio_data, expected_rate)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(resampled)

    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Downsample WAV files to 8 kHz using soxr."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        help="Directory containing WAV files to downsample.",
    )
    source.add_argument(
        "--files",
        nargs="+",
        metavar="WAV",
        help="Specific WAV files to downsample.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for downsampled files.",
    )
    parser.add_argument(
        "--speaker",
        choices=["patient", "doctor", "both"],
        default="patient",
        help="Which speaker files to process when using --input-dir "
        "(filters by *_patient.wav / *_doctor.wav). Default: patient.",
    )
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=16000,
        help="Expected input sample rate in Hz. Default: 16000.",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.files:
        wav_files = [f for f in args.files if f.endswith(".wav") and os.path.exists(f)]
        if len(wav_files) < len(args.files):
            skipped = len(args.files) - len(wav_files)
            print(f"Warning: skipped {skipped} file(s) (not found or not .wav)")
    else:
        patterns = []
        if args.speaker in ("patient", "both"):
            patterns.append(os.path.join(args.input_dir, "*_patient.wav"))
        if args.speaker in ("doctor", "both"):
            patterns.append(os.path.join(args.input_dir, "*_doctor.wav"))
        wav_files = sorted(p for pat in patterns for p in glob(pat))

    print(f"Processing {len(wav_files)} file(s)")

    ok = 0
    for path in wav_files:
        name = Path(path).stem
        out = os.path.join(args.output_dir, f"{name}_8kHz.wav")
        print(f"  {Path(path).name} -> {Path(out).name}")
        try:
            if downsample_file(path, out, args.input_sample_rate):
                ok += 1
        except Exception as exc:
            print(f"  Error: {exc}")

    print(f"Done — {ok}/{len(wav_files)} file(s) downsampled to {TARGET_SAMPLE_RATE} Hz")


if __name__ == "__main__":
    main()
