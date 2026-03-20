"""Compute psychoacoustic building-block metrics for the Clotho dataset (Zenodo 3490684) - OPTIMIZED.

Dataset
-------
Zenodo record: https://zenodo.org/record/3490684
Clotho is an audio captioning dataset with 15-30s WAV clips and 5 captions per clip.

This script treats the dataset as two splits:
- development: clotho_captions_development.csv + audio_development/development/*.wav
- evaluation:  clotho_captions_evaluation.csv  + audio_evaluation/evaluation/*.wav

Output
------
Chunked output directory (recommended):
  <out>/parts/part-*.parquet
  <out>/done_ids.txt
  <out>/provenance.json

Each row includes:
- id: "<split>/<file_name>"
- audio_path
- text: we use caption_1 as the canonical text
- caption_1..caption_5 (all preserved)
- metrics: loudness, sharpness, roughness (optional), fluctuation proxy (optional), EVR/AMI/RMS_mod, tonality PR (optional)

Notes
-----
MOSQITO expects calibrated Pa for absolute SPL; these waveforms are uncalibrated.
Metrics are still useful as relative features.

Design
------
This follows the same durable, resumable, chunked approach used for DataSEC:
- --resume skips ids already in done_ids.txt
- --max-new limits work per invocation to avoid losing progress

Typical usage (via run_metrics_batch.sh):
  python scripts/clotho_3490684_compute_metrics_v2.py \
    --clotho-root /root/clawd/_datasets/clotho_3490684 \
    --out /root/clawd/_datasets/clotho_3490684/metrics \
    --n-jobs 3 --batch-size 20 --chunk 100 --resume --max-new 200 --skip-fluctuation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt
from tqdm import tqdm

from mosqito.sq_metrics import loudness_zwst, sharpness_din_st, roughness_dw, loudness_zwtv
from mosqito.sq_metrics.tonality import pr_ecma_st


def load_wav(path: Path, max_duration_sec: Optional[float] = None) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)

    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(max_duration_sec * sr)
        if len(x) > max_samples:
            x = x[:max_samples]

    return int(sr), x


def fluctuation_strength_proxy(x: np.ndarray, sr: int) -> float:
    """Expensive proxy FS from loudness time series (0.5–20 Hz RMS band)."""
    N_t, _, _, t = loudness_zwtv(x, sr)
    if len(t) < 4:
        return 0.0
    fs_loud = 1.0 / float(np.mean(np.diff(t)))

    lo, hi = 0.5, 20.0
    nyq = 0.5 * fs_loud
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        return 0.0

    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    from scipy.signal import filtfilt

    y = filtfilt(b, a, N_t)
    return float(np.sqrt(np.mean(y**2)))


def envelope_modulation_proxies(x: np.ndarray, sr: int, fmin: float = 0.5, fmax: float = 20.0) -> tuple[float, float, float]:
    """Cheap temporal-modulation proxies (substitutes for fluctuation strength).

    Returns (EVR, AMI, RMS_mod).
    """

    if x.size < 8:
        return float("nan"), float("nan"), float("nan")

    env = np.abs(hilbert(x.astype(np.float64)))

    nyq = 0.5 * sr
    hi = min(fmax, 0.99 * nyq)
    if hi <= fmin:
        return float("nan"), float("nan"), float("nan")

    sos = butter(2, [fmin / nyq, hi / nyq], btype="band", output="sos")
    env_mod = sosfiltfilt(sos, env)

    m = float(np.mean(env_mod))
    v = float(np.var(env_mod))
    eps = 1e-10

    evr = v / (m * m + eps)
    ami = (float(np.max(env_mod)) - float(np.min(env_mod))) / (abs(m) + eps)
    rms_mod = float(np.sqrt(np.mean(env_mod * env_mod)))

    return float(evr), float(ami), float(rms_mod)


def tonality_scalar_pr(x: np.ndarray, sr: int) -> float:
    pr = pr_ecma_st(x, sr)
    if isinstance(pr, tuple):
        pr = pr[0]
    if isinstance(pr, np.ndarray):
        return float(np.max(pr)) if pr.size else 0.0
    return float(pr)


def compute_row(
    wav_path: Path,
    wav_id: str,
    text: str,
    captions: dict,
    skip_roughness: bool,
    skip_fluctuation: bool,
    skip_tonality: bool,
    max_duration_sec: Optional[float],
) -> dict:
    start_time = time.time()

    sr, x = load_wav(wav_path, max_duration_sec=max_duration_sec)

    loud = float(loudness_zwst(x, sr)[0])
    sharp = float(sharpness_din_st(x, sr))
    evr, ami, rms_mod = envelope_modulation_proxies(x, sr)

    rough = float("nan")
    fs_proxy = float("nan")
    pr = float("nan")

    if not skip_roughness:
        r = roughness_dw(x, sr, overlap=0)[0]
        rough = float(np.max(r)) if isinstance(r, np.ndarray) else float(r)

    if not skip_fluctuation:
        fs_proxy = fluctuation_strength_proxy(x, sr)

    if not skip_tonality:
        pr = tonality_scalar_pr(x, sr)

    return {
        "id": wav_id,
        "audio_path": str(wav_path),
        "text": text,
        **captions,
        "sample_rate": sr,
        "duration_s": float(len(x) / sr),
        "loudness": loud,
        "sharpness": sharp,
        "roughness": rough,
        "fluctuation_strength_proxy": fs_proxy,
        "pr_ecma_st": pr,
        "fluctuation_evr": evr,
        "fluctuation_ami": ami,
        "fluctuation_rms_mod": rms_mod,
        "error": "",
        "compute_time_s": round(time.time() - start_time, 3),
    }


def load_manifest(clotho_root: Path) -> pd.DataFrame:
    """Return a unified manifest with split, file_name, captions, and wav_path."""

    dev_caps = pd.read_csv(clotho_root / "clotho_captions_development.csv")
    dev_caps["split"] = "development"
    eval_caps = pd.read_csv(clotho_root / "clotho_captions_evaluation.csv")
    eval_caps["split"] = "evaluation"

    df = pd.concat([dev_caps, eval_caps], ignore_index=True)

    dev_audio_root = clotho_root / "audio_development" / "development"
    eval_audio_root = clotho_root / "audio_evaluation" / "evaluation"

    def to_path(row) -> Path:
        if row["split"] == "development":
            return dev_audio_root / row["file_name"]
        return eval_audio_root / row["file_name"]

    df["wav_path"] = df.apply(to_path, axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute psychoacoustic metrics for Clotho (Zenodo 3490684) - OPTIMIZED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--clotho-root", required=True, help="Path to Clotho root dir (contains CSVs and audio_*/)")
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output path (.parquet/.csv) OR an output directory (recommended) for chunked writes. "
            "If a directory is provided, this script will write multiple part-*.parquet files and can resume."
        ),
    )

    ap.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Number of parallel jobs (default: {cpu_count()} - 1)",
    )
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=None)

    ap.add_argument("--skip-roughness", action="store_true")
    ap.add_argument("--skip-fluctuation", action="store_true")
    ap.add_argument("--skip-tonality", action="store_true")
    ap.add_argument("--max-duration-seconds", type=float, default=None)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    clotho_root = Path(args.clotho_root)

    df = load_manifest(clotho_root)
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    # Setup output (dir mode recommended)
    out_target = Path(args.out)
    out_is_dir = out_target.suffix == ""
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
        "dataset": "Clotho",
        "zenodo_record": "https://zenodo.org/record/3490684",
        "clotho_root": str(clotho_root),
        "mosqito": "1.2.1",
        "note": "Metrics computed on uncalibrated waveforms; absolute values depend on SPL calibration.",
        "script_version": "v2-optimized",
        "metrics_config": {
            "skip_roughness": bool(args.skip_roughness),
            "skip_fluctuation": bool(args.skip_fluctuation),
            "skip_tonality": bool(args.skip_tonality),
            "max_duration_seconds": args.max_duration_seconds,
        },
        "performance_config": {
            "n_jobs": args.n_jobs,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk,
        },
    }

    # Build list of items to process
    items = []
    for _, r in df.iterrows():
        file_name = str(r["file_name"])
        split = str(r["split"])
        wav_id = f"{split}/{file_name}"
        wav_path = Path(r["wav_path"])
        if out_is_dir and args.resume and wav_id in done_ids:
            continue
        items.append((wav_path, wav_id, r))

    if args.max_new is not None and len(items) > args.max_new:
        items = items[: int(args.max_new)]

    if not items:
        print("No files to process. Exiting.")
        return

    buf: list[dict] = []
    wrote = 0

    # IMPORTANT: when resuming in directory-output mode, do NOT overwrite existing parts.
    # Continue part numbering from what already exists.
    part_idx = 0
    if out_is_dir:
        assert parts_dir is not None
        existing = sorted(parts_dir.glob("part-*.parquet"))
        if existing:
            part_idx = len(existing)
            print(f"Resuming parts: found {part_idx} existing part parquet files; will continue numbering.")

    total_start = time.time()

    def flush_chunk() -> None:
        nonlocal buf, part_idx
        if not buf:
            return
        df_chunk = pd.DataFrame(buf)
        assert parts_dir is not None
        part_path = parts_dir / f"part-{part_idx:05d}.parquet"
        df_chunk.to_parquet(part_path, index=False)
        print(f"Wrote part-{part_idx:05d}.parquet ({len(df_chunk)} rows)")
        part_idx += 1
        buf = []

    n_batches = (len(items) + args.batch_size - 1) // args.batch_size

    for batch_i in tqdm(range(n_batches), desc="Processing batches", unit="batch"):
        b0 = batch_i * args.batch_size
        b1 = min(b0 + args.batch_size, len(items))
        batch = items[b0:b1]

        def one(it):
            wav_path, wav_id, r = it
            captions = {
                "caption_1": str(r.get("caption_1", "")),
                "caption_2": str(r.get("caption_2", "")),
                "caption_3": str(r.get("caption_3", "")),
                "caption_4": str(r.get("caption_4", "")),
                "caption_5": str(r.get("caption_5", "")),
                "split": str(r.get("split", "")),
                "file_name": str(r.get("file_name", "")),
            }
            text = captions["caption_1"]

            if not wav_path.exists():
                return {
                    "id": wav_id,
                    "audio_path": str(wav_path),
                    "text": text,
                    **captions,
                    "sample_rate": None,
                    "duration_s": None,
                    "loudness": float("nan"),
                    "sharpness": float("nan"),
                    "roughness": float("nan"),
                    "fluctuation_strength_proxy": float("nan"),
                    "pr_ecma_st": float("nan"),
                    "fluctuation_evr": float("nan"),
                    "fluctuation_ami": float("nan"),
                    "fluctuation_rms_mod": float("nan"),
                    "error": "FileNotFoundError",
                    "compute_time_s": 0.0,
                }

            try:
                return compute_row(
                    wav_path=wav_path,
                    wav_id=wav_id,
                    text=text,
                    captions=captions,
                    skip_roughness=bool(args.skip_roughness),
                    skip_fluctuation=bool(args.skip_fluctuation),
                    skip_tonality=bool(args.skip_tonality),
                    max_duration_sec=args.max_duration_seconds,
                )
            except Exception as e:
                out = {
                    "id": wav_id,
                    "audio_path": str(wav_path),
                    "text": text,
                    **captions,
                    "sample_rate": None,
                    "duration_s": None,
                    "loudness": float("nan"),
                    "sharpness": float("nan"),
                    "roughness": float("nan"),
                    "fluctuation_strength_proxy": float("nan"),
                    "pr_ecma_st": float("nan"),
                    "fluctuation_evr": float("nan"),
                    "fluctuation_ami": float("nan"),
                    "fluctuation_rms_mod": float("nan"),
                    "error": repr(e),
                    "compute_time_s": 0.0,
                }
                if args.fail_fast:
                    raise
                return out

        if args.n_jobs == 1:
            batch_results = [one(it) for it in batch]
        else:
            batch_results = Parallel(
                n_jobs=args.n_jobs,
                verbose=0,
                backend="loky",
                prefer="processes",
            )(delayed(one)(it) for it in batch)

        buf.extend(batch_results)
        wrote += len(batch_results)

        if out_is_dir:
            assert done_ids_path is not None
            with done_ids_path.open("a", encoding="utf-8") as f:
                for rr in batch_results:
                    f.write(str(rr["id"]) + "\n")

            if len(buf) >= args.chunk:
                flush_chunk()

        if (batch_i + 1) % 5 == 0 or batch_i == n_batches - 1:
            elapsed = time.time() - total_start
            fps = wrote / elapsed if elapsed > 0 else 0.0
            print(f"Progress: {wrote}/{len(items)} | {fps:.2f} files/s")

    if out_is_dir:
        flush_chunk()
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Completed! Wrote ~{wrote} rows into {out_dir}/parts")
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
        print(f"Completed! Wrote {len(df_out)} rows to {out_target}")


if __name__ == "__main__":
    main()
