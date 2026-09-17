"""Recover observed structure from a synthetic, denormalized laboratory table.

Run: uv run python examples/wide_table.py /tmp/fieldwork-wide-table
The table has 960 rows and 43 columns, with three technical replicates per
specimen/assay. No schema, edges, or attribute assignments are supplied to grain().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw


def sample() -> pd.DataFrame:
    """Flatten participant, specimen, assay, instrument, run and result records."""
    rng = np.random.default_rng(2026)
    instruments = [
        {
            "instrument_id": f"I{i + 1}",
            "instrument_model": f"Reader {i + 1}",
            "instrument_vendor": ["Acme", "Beacon", "Cedar"][i],
            "lab_room": f"Lab {i + 1}",
            "installation_year": 2020 + i,
            "service_interval_days": [90, 120, 180][i],
            "firmware_version": f"{i + 2}.1",
        }
        for i in range(3)
    ]
    assays = [
        {
            "assay_id": f"A{a + 1}",
            "analyte": ["marker_alpha", "marker_beta", "marker_gamma", "marker_delta"][a],
            "assay_method": ["fluorescence", "luminescence", "absorbance", "fluorescence"][a],
            "result_unit": ["ng/mL", "pg/mL", "U/L", "mg/L"][a],
            "detection_limit": [0.1, 0.2, 0.5, 0.05][a],
            "reference_upper": [20.0, 50.0, 100.0, 10.0][a],
            "dilution_factor": [2, 5, 10, 20][a],
        }
        for a in range(4)
    ]
    rows = []
    for p in range(40):
        participant = {
            "participant_id": f"P{p + 1:03}",
            "enrollment_site": ["North", "South", "East", "West"][p % 4],
            "cohort": ["discovery", "validation"][p % 2],
            "age_at_enrollment": int(rng.integers(25, 75)),
            "study_arm": ["control", "intervention"][(p // 2) % 2],
            "enrollment_month": f"2025-{p % 12 + 1:02}",
            "smoking_status": ["never", "former", "current"][p % 3],
        }
        for visit in range(2):
            specimen = {
                "specimen_id": f"S{p * 2 + visit + 1:03}",
                "visit_number": visit + 1,
                "collection_day": p * 3 + visit * 90,
                "sample_volume_ml": round(float(rng.uniform(1.0, 5.0)), 2),
                "storage_hours": int(rng.integers(2, 48)),
                "freeze_thaw_cycles": int(rng.integers(0, 3)),
                "hemolysis_index": round(float(rng.uniform(0, 0.5)), 3),
            }
            for a, assay in enumerate(assays):
                # Ten participants share a batch. Instruments serve several assays;
                # each specimen is measured in four runs, one per assay.
                batch = p // 10
                run_number = (batch * 2 + visit) * 4 + a
                run = {
                    "run_id": f"R{run_number + 1:03}",
                    "run_day": batch * 7 + visit * 90 + a,
                    "operator": f"operator_{run_number % 5 + 1}",
                    "reagent_lot": f"lot_{run_number % 7 + 1}",
                    "calibration_factor": round(0.95 + run_number * 0.003, 3),
                    "ambient_temp_c": round(20 + (run_number % 9) * 0.2, 1),
                    "run_qc_status": "review" if run_number % 11 == 0 else "pass",
                }
                concentration = round(float(rng.uniform(1, 80)), 3)
                measurement = {
                    "concentration": concentration,
                    "normalized_signal": round(concentration / assay["dilution_factor"], 3),
                    "result_flag": "high" if concentration > assay["reference_upper"] else "normal",
                    "recovery_pct": round(float(rng.uniform(85, 115)), 2),
                    "replicate_cv_pct": round(float(rng.uniform(0.1, 8)), 2),
                    "review_status": str(rng.choice(["accepted", "pending", "repeat"])),
                }
                for replicate in range(3):
                    rows.append(
                        {
                            **participant,
                            **specimen,
                            **assay,
                            **instruments[(batch + visit + a) % 3],
                            **run,
                            **measurement,
                            "replicate": replicate + 1,
                            "raw_signal": round(concentration + float(rng.normal(0, 0.1)), 4),
                        }
                    )
    return pd.DataFrame(rows)


def investigate():
    df = sample()
    graph = fw.grain(
        df,
        candidate_keys=[
            "participant_id",
            "instrument_id",
            "assay_id",
            "specimen_id",
            "run_id",
            fw.KeySpec("specimen_assay", ("specimen_id", "assay_id")),
        ],
    )
    return df, graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df, graph = investigate()
    (args.output / "wide-table.svg").write_text(fw.render_svg(graph), encoding="utf-8")
    (args.output / "wide-table.html").write_text(fw.render_html(graph), encoding="utf-8")
    print(f"{len(df):,} rows × {len(df.columns)} columns; figures in {args.output.resolve()}")
