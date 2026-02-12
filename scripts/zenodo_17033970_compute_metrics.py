"""Compute psychoacoustic building-block metrics for DataSEC (Zenodo 17033970).

Dataset
-------
Zenodo record: https://zenodo.org/records/17033970
Title: "DataSEC - Dataset for Sound Event Classification of environmental noise"

Observed layout after unzip:
  <dataset-root>/Bells/*.wav
  <dataset-root>/Crows seagulls and magpies/Crows/*.wav
  ...

So we treat the *top-level* directory name under dataset-root as the class label,
and (optionally) the second-level as a sublabel.

Output
------
We record building-block psychoacoustic metrics (not a single annoyance score):
- loudness (Zwicker stationary)
- sharpness (DIN)
- roughness (Daniel & Weber): we take max over time
- fluctuation strength proxy (from bandpassed time-varying loudness modulation)
- tonality building block: prominence ratio (ECMA 418-1)

Notes
-----
MOSQITO expects calibrated Pa for absolute SPL. These waveforms are uncalibrated;
metrics are still useful as relative features for learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

from mosqito.sq_metrics import loudness_zwst, sharpness_din_st, roughness_dw, loudness_zwtv
from mosqito.sq_metrics.tonality import pr_ecma_st


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return int(sr), x


def fluctuation_strength_proxy(x: np.ndarray, sr: int) -> float:
    N_t, _, _, t = loudness_zwtv(x, sr)
    if len(t) < 4:
        return 0.0
    fs_loud = 1.0 / float(np.mean(np.diff(t)))

    from scipy.signal import butter, filtfilt

    lo, hi = 0.5, 20.0
    nyq = 0.5 * fs_loud
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        return 0.0

    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    y = filtfilt(b, a, N_t)
    return float(np.sqrt(np.mean(y**2)))


def tonality_scalar_pr(x: np.ndarray, sr: int) -> float:
    pr = pr_ecma_st(x, sr)
    if isinstance(pr, tuple):
        pr = pr[0]
    if isinstance(pr, np.ndarray):
        return float(np.max(pr)) if pr.size else 0.0
    return float(pr)


def compute_row(wav_path: Path) -> dict:
    sr, x = load_wav(wav_path)

    loud = float(loudness_zwst(x, sr)[0])
    sharp = float(sharpness_din_st(x, sr))

    r = roughness_dw(x, sr, overlap=0)[0]
    rough = float(np.max(r)) if isinstance(r, np.ndarray) else float(r)

    fs_proxy = fluctuation_strength_proxy(x, sr)
    pr = tonality_scalar_pr(x, sr)

    return {
        "sample_rate": sr,
        "duration_s": float(len(x) / sr),
        "loudness": loud,
        "sharpness": sharp,
        "roughness": rough,
        "fluctuation_strength_proxy": fs_proxy,
        "pr_ecma_st": pr,
    }


def iter_wavs(dataset_root: Path) -> list[Path]:
    wavs: list[Path] = []
    for p in dataset_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".wav":
            wavs.append(p)
    wavs.sort()
    return wavs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Path to extracted DATASEC audio root")
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output path (.parquet/.csv) OR an output directory (recommended) for chunked writes. "
            "If a directory is provided, this script will write multiple part-*.parquet files and can resume."
        ),
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    ap.add_argument("--chunk", type=int, default=100, help="Rows per parquet part when using directory output")
    ap.add_argument("--resume", action="store_true", help="Resume by skipping ids already present in output")
    ap.add_argument(
        "--max-new",
        type=int,
        default=None,
        help=(
            "Process at most this many *new* files (after resume-skips) and then exit cleanly. "
            "Useful to avoid long runs getting SIGKILL'd by restarting in batches."
        ),
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    wavs = iter_wavs(dataset_root)
    if args.limit is not None:
        wavs = wavs[: int(args.limit)]

    out_target = Path(args.out)
    out_is_dir = out_target.suffix == ""  # no extension => treat as directory
    if out_is_dir:
        out_dir = out_target
        out_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = out_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        done_ids_path = out_dir / "done_ids.txt"
        done_ids: set[str] = set()
        if args.resume and done_ids_path.exists():
            done_ids = set(done_ids_path.read_text(encoding="utf-8").splitlines())
    else:
        out_dir = out_target.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = None
        done_ids_path = None
        done_ids = set()

    provenance = {
        "dataset": "DataSEC",
        "zenodo_record": "https://zenodo.org/records/17033970",
        "dataset_root": str(dataset_root),
        "mosqito": "1.2.1",
        "note": "Metrics computed on uncalibrated waveforms; absolute values depend on SPL calibration.",
    }

    buf: list[dict] = []
    wrote = 0
    part_idx = 0
    new_processed = 0

    def flush_chunk() -> None:
        nonlocal buf, wrote, part_idx
        if not buf:
            return
        df_chunk = pd.DataFrame(buf)
        if out_is_dir:
            assert parts_dir is not None
            part_path = parts_dir / f"part-{part_idx:05d}.parquet"
            df_chunk.to_parquet(part_path, index=False)
            part_idx += 1
        buf = []

    for wav_path in tqdm(wavs, desc="DataSEC metrics"):
        rel = wav_path.relative_to(dataset_root)
        wav_id = rel.as_posix()

        if out_is_dir and args.resume and wav_id in done_ids:
            continue
        if args.max_new is not None and new_processed >= int(args.max_new):
            break

        parts = rel.parts
        label = parts[0] if len(parts) >= 2 else ""
        sublabel = parts[1] if len(parts) >= 3 else ""

        base = {
            "id": wav_id,
            "audio_path": str(wav_path),
            "text": label,  # dataset appears label-only; use label as placeholder text
            "label": label,
            "sublabel": sublabel,
            "error": "",
        }

        try:
            metrics = compute_row(wav_path)
            base.update(metrics)
        except Exception as e:
            base["error"] = repr(e)
            if args.fail_fast:
                raise

        buf.append(base)
        wrote += 1
        new_processed += 1

        if out_is_dir:
            assert done_ids_path is not None
            with done_ids_path.open("a", encoding="utf-8") as f:
                f.write(wav_id + "\n")

        if out_is_dir and len(buf) >= int(args.chunk):
            flush_chunk()

    if out_is_dir:
        flush_chunk()
        prov_path = out_dir / "provenance.json"
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Wrote ~{wrote} rows into {parts_dir} (chunked)")
    else:
        df_out = pd.DataFrame(buf)
        if out_target.suffix.lower() == ".parquet":
            df_out.to_parquet(out_target, index=False)
        elif out_target.suffix.lower() == ".csv":
            df_out.to_csv(out_target, index=False)
        else:
            raise ValueError("--out must end with .parquet or .csv")
        prov_path = out_target.with_suffix(out_target.suffix + ".provenance.json")
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Wrote {len(df_out)} rows to {out_target}")


if __name__ == "__main__":
    main()
