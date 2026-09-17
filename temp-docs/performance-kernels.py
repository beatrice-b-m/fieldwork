"""Historical audit prototype; run against baseline d0a4d9f, not current code.

Audit-only kernel experiment, not a proposed public API or production patch.

Run after (not concurrently with) benchmarks/scaling.py. Verifies exact fingerprint
equality and row-presence array equality on three seeded synthetic fixtures.
Does not establish parity for all supported scalar/index types or whole operations.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import time
from pathlib import Path

import numpy as np

from fieldwork._explore.encoding import encode_series, normalize_scalar
from fieldwork.evidence import fingerprint


def dictionary_fingerprint(frame):
    digest = hashlib.sha256()
    for labels in (frame.columns, frame.index):
        digest.update(b"[")
        for value in labels:
            digest.update(
                json.dumps(
                    normalize_scalar(value, label=isinstance(value, tuple)).to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
        digest.update(b"]")
    for column in frame:
        values, codes = encode_series(frame[column])
        serialized = np.array(
            [
                json.dumps(value.to_dict(), sort_keys=True, allow_nan=False).encode() + b"\n"
                for value in values
            ],
            dtype=object,
        )
        digest.update(b"[")
        for start in range(0, len(codes), 8192):
            digest.update(b"".join(serialized[codes[start : start + 8192]].tolist()))
        digest.update(b"]")
    return digest.hexdigest()


def timed(run, *args):
    start = time.perf_counter()
    output = run(*args)
    return output, time.perf_counter() - start


def original_masks(present, rows):
    units = [[i] for i in range(rows)]
    return {
        c: np.array([np.any(mask[indices]) for indices in units], dtype=bool)
        for c, mask in present.items()
    }


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = runpy.run_path(str(root / "benchmarks/scaling.py"))["fixture"]
    records = []
    for kind in ("sparse", "dense", "mixed"):
        frame = fixture(10_000, 150, kind, 721)
        original, original_seconds = timed(fingerprint, frame)
        proposed, proposed_seconds = timed(dictionary_fingerprint, frame)
        if original != proposed:
            raise AssertionError(f"Fingerprint mismatch for {kind}")
        present = {c: frame[c].notna().to_numpy() for c in frame}

        original_arrays, mask_seconds = timed(original_masks, present, len(frame))
        proposed_arrays, direct_seconds = timed(dict, present)
        if not all(np.array_equal(original_arrays[c], proposed_arrays[c]) for c in present):
            raise AssertionError(f"Row mask mismatch for {kind}")
        records.append(
            {
                "fixture": kind,
                "rows": len(frame),
                "columns": len(frame.columns),
                "fingerprint": {
                    "original_seconds": original_seconds,
                    "dictionary_seconds": proposed_seconds,
                    "identical_sha256": True,
                },
                "row_masks": {
                    "original_seconds": mask_seconds,
                    "reuse_seconds": direct_seconds,
                    "identical_arrays": True,
                },
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
