"""Compute psychoacoustic building-block metrics for DataSEC (Zenodo 17033970) - OPTIMIZED VERSION.

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

Performance Improvements (v2)
-----------------------------
- Parallel processing with joblib (multiprocessing)
- Optional metric skipping via flags
- Audio duration limiting (--max-duration-seconds)
- Better progress logging with timing and ETA
- Batching to avoid SIGKILL from long runs
- Configurable chunk size and parallelism

Usage Examples
--------------
# Full metrics, parallel, resume
python scripts/zenodo_17033970_compute_metrics_v2.py \
  --dataset-root /root/clawd/_datasets/zenodo_17033970/ \
  --out /root/clawd/_datasets/zenodo_17033970/metrics \
  --n-jobs 4 \
  --chunk 100 \
  --max-new 500 \
  --resume

# Fast mode: skip expensive metrics, limit duration
python scripts/zenodo_17033970_compute_metrics_v2.py \
  --dataset-root /root/clawd/_datasets/zenodo_17033970/ \
  --out /root/clawd/_datasets/zenodo_17033970/metrics \
  --skip-roughness \
  --skip-fluctuation \
  --skip-tonality \
  --max-duration-seconds 10

# Debug: limit files, serial, fast metrics only
python scripts/zenodo_17033970_compute_metrics_v2.py \
  --dataset-root /root/clawd/_datasets/zenodo_17033970/ \
  --out debug_output.parquet \
  --limit 10 \
  --skip-roughness \
  --skip-fluctuation \
  --n-jobs 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Import mosqito metrics - these are imported at module level for worker processes
from mosqito.sq_metrics import loudness_zwst, sharpness_din_st, roughness_dw, loudness_zwtv
from mosqito.sq_metrics.tonality import pr_ecma_st


def load_wav(path: Path, max_duration_sec: Optional[float] = None) -> tuple[int, np.ndarray]:
    """Load WAV file with optional duration truncation."""
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Optionally truncate audio
    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(max_duration_sec * sr)
        if len(x) > max_samples:
            x = x[:max_samples]

    return int(sr), x


def fluctuation_strength_proxy(x: np.ndarray, sr: int) -> float:
    """Compute fluctuation strength proxy from time-varying loudness.

    NOTE: This relies on mosqito.loudness_zwtv and is very expensive.
    In our DataSEC runs we typically skip it via --skip-fluctuation.
    """

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

    # keep the original filtfilt behavior without adding imports at module import time
    from scipy.signal import filtfilt

    y = filtfilt(b, a, N_t)
    return float(np.sqrt(np.mean(y**2)))


def envelope_modulation_proxies(x: np.ndarray, sr: int, fmin: float = 0.5, fmax: float = 20.0) -> tuple[float, float, float]:
    """Cheap temporal-modulation proxies (as substitute for fluctuation strength).

    Computes amplitude-envelope modulation in the fluctuation-sensitive band (0.5–20 Hz):
    - EVR: envelope variance ratio = var(env_mod) / (mean(env_mod)^2 + eps)
    - AMI: amplitude modulation index = (max-min) / (mean + eps)
    - RMS_mod: RMS of env_mod

    Uses Hilbert envelope + 2nd-order Butterworth bandpass on the envelope.
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
    """Compute tonality scalar from Prominence Ratio (ECMA 418-1)."""
    pr = pr_ecma_st(x, sr)
    if isinstance(pr, tuple):
        pr = pr[0]
    if isinstance(pr, np.ndarray):
        return float(np.max(pr)) if pr.size else 0.0
    return float(pr)


# EVR temporal modulation proxy functions (cheap, no MOSQITO dependency)


def evr_temporal_modulation(x: np.ndarray, sr: int, lo: float = 0.5, hi: float = 20.0) -> float:
    """Compute Envelope Variance Ratio (EVR) as a temporal modulation proxy.

    EVR is computed from the waveform envelope bandpassed in the 0.5-20 Hz range.
    This is a cheap substitute for fluctuation strength when MOSQITO is skipped.

    Args:
        x: Audio signal (normalized float64)
        sr: Sample rate
        lo: Lower bandpass cutoff frequency in Hz (default: 0.5)
        hi: Upper bandpass cutoff frequency in Hz (default: 20.0)

    Returns:
        EVR value (variance of bandpassed envelope / variance of original envelope)
        Returns 0.0 on error or invalid input.
    """
    if len(x) < 4:
        return 0.0

    nyq = 0.5 * sr
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        return 0.0

    try:
        # Compute analytic signal and extract envelope
        analytic = hilbert(x.astype(np.float64))
        envelope = np.abs(analytic)

        # Compute variance of original envelope
        var_env = np.var(envelope)
        if var_env < 1e-20:  # Prevent division by zero
            return 0.0

        # Bandpass filter the envelope (0.5-20 Hz)
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        envelope_bp = sosfiltfilt(sos, envelope)

        # Compute variance of bandpassed envelope
        var_env_bp = np.var(envelope_bp)

        # EVR: variance ratio
        evr = var_env_bp / var_env
        return float(evr)

    except Exception:
        return 0.0


def ami_temporal_modulation(x: np.ndarray, sr: int, lo: float = 0.5, hi: float = 20.0) -> float:
    """Compute Amplitude Modulation Index (AMI) as a temporal modulation feature.

    AMI measures the normalized variance of the bandpassed envelope.

    Args:
        x: Audio signal (normalized float64)
        sr: Sample rate
        lo: Lower bandpass cutoff frequency in Hz (default: 0.5)
        hi: Upper bandpass cutoff frequency in Hz (default: 20.0)

    Returns:
        AMI value (0-1 range). Returns NaN on error.
    """
    if len(x) < 4:
        return float("nan")

    nyq = 0.5 * sr
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        return float("nan")

    try:
        # Compute analytic signal and extract envelope
        analytic = hilbert(x.astype(np.float64))
        envelope = np.abs(analytic)

        # Bandpass filter the envelope
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        envelope_bp = sosfiltfilt(sos, envelope)

        # AMI: normalized variance (std / mean)
        mean_bp = np.mean(envelope_bp)
        if mean_bp < 1e-20:
            return float("nan")
        ami = np.std(envelope_bp) / mean_bp
        return float(ami)

    except Exception:
        return float("nan")


def rms_modulation(x: np.ndarray, sr: int, lo: float = 0.5, hi: float = 20.0) -> float:
    """Compute RMS Modulation as a temporal modulation feature.

    RMS modulation measures the RMS of the bandpassed envelope relative to mean envelope.

    Args:
        x: Audio signal (normalized float64)
        sr: Sample rate
        lo: Lower bandpass cutoff frequency in Hz (default: 0.5)
        hi: Upper bandpass cutoff frequency in Hz (default: 20.0)

    Returns:
        RMS modulation value. Returns NaN on error.
    """
    if len(x) < 4:
        return float("nan")

    nyq = 0.5 * sr
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        return float("nan")

    try:
        # Compute analytic signal and extract envelope
        analytic = hilbert(x.astype(np.float64))
        envelope = np.abs(analytic)

        # Bandpass filter the envelope
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        envelope_bp = sosfiltfilt(sos, envelope)

        # RMS modulation: RMS of bandpassed envelope / mean of original envelope
        mean_env = np.mean(envelope)
        if mean_env < 1e-20:
            return float("nan")
        rms_bp = np.sqrt(np.mean(envelope_bp**2))
        rms_mod = rms_bp / mean_env
        return float(rms_mod)

    except Exception:
        return float("nan")


def envelope_modulation_proxies(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Compute all envelope-based temporal modulation proxies.

    This is a convenience wrapper that returns EVR, AMI, and RMS modulation.

    Args:
        x: Audio signal (normalized float64)
        sr: Sample rate

    Returns:
        Tuple of (evr, ami, rms_mod) - all three temporal modulation proxies.
    """
    evr = evr_temporal_modulation(x, sr)
    ami = ami_temporal_modulation(x, sr)
    rms_mod = rms_modulation(x, sr)
    return evr, ami, rms_mod


def compute_row(
    wav_path: Path,
    dataset_root: Path,
    skip_roughness: bool = False,
    skip_fluctuation: bool = False,
    skip_tonality: bool = False,
    max_duration_sec: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """Compute metrics for a single WAV file.

    Args:
        wav_path: Path to WAV file
        dataset_root: Root dataset directory for relative path computation
        skip_roughness: Skip roughness_dw computation
        skip_fluctuation: Skip fluctuation_strength_proxy computation
        skip_tonality: Skip pr_ecma_st computation
        max_duration_sec: Optional duration limit in seconds
        verbose: Enable debug logging

    Returns:
        Dictionary with all computed fields (including skipped metrics as NaN)
    """
    start_time = time.time()

    # Load audio
    sr, x = load_wav(wav_path, max_duration_sec=max_duration_sec)
    load_time = time.time() - start_time

    # Always compute these (fast)
    loud = float(loudness_zwst(x, sr)[0])
    sharp = float(sharpness_din_st(x, sr))

    # Cheap temporal-modulation proxies (recommended substitute when skipping fluctuation)
    evr, ami, rms_mod = envelope_modulation_proxies(x, sr)

    # Optional expensive metrics
    rough = float("nan")
    fs_proxy = float("nan")
    pr = float("nan")

    if not skip_roughness:
        r_start = time.time()
        r = roughness_dw(x, sr, overlap=0)[0]
        rough = float(np.max(r)) if isinstance(r, np.ndarray) else float(r)
        if verbose:
            print(f"  roughness_dw: {time.time() - r_start:.2f}s")

    if not skip_fluctuation:
        f_start = time.time()
        fs_proxy = fluctuation_strength_proxy(x, sr)
        if verbose:
            print(f"  fluctuation_strength_proxy: {time.time() - f_start:.2f}s")

    if not skip_tonality:
        t_start = time.time()
        pr = tonality_scalar_pr(x, sr)
        if verbose:
            print(f"  pr_ecma_st: {time.time() - t_start:.2f}s")

    # Compute relative path
    rel = wav_path.relative_to(dataset_root)
    parts = rel.parts
    label = parts[0] if len(parts) >= 2 else ""
    sublabel = parts[1] if len(parts) >= 3 else ""

    total_time = time.time() - start_time

    return {
        "id": rel.as_posix(),
        "audio_path": str(wav_path),
        "text": label,  # dataset appears label-only; use label as placeholder text
        "label": label,
        "sublabel": sublabel,
        "sample_rate": sr,
        "duration_s": float(len(x) / sr),
        "loudness": loud,
        "sharpness": sharp,
        "roughness": rough,
        "fluctuation_strength_proxy": fs_proxy,
        "pr_ecma_st": pr,
        # fluctuation substitute features (fast):
        "fluctuation_evr": evr,
        "fluctuation_ami": ami,
        "fluctuation_rms_mod": rms_mod,
        "error": "",
        "compute_time_s": round(total_time, 3),
    }


def iter_wavs(dataset_root: Path) -> list[Path]:
    """Get sorted list of all WAV files in dataset."""
    wavs: list[Path] = []
    for p in dataset_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".wav":
            wavs.append(p)
    wavs.sort()
    return wavs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute psychoacoustic metrics for DataSEC (Zenodo 17033970) - OPTIMIZED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full parallel run with resume and batching
  %(prog)s --dataset-root /path/to/data --out /path/to/output --n-jobs 4 --max-new 500 --resume

  # Fast mode: skip expensive metrics, limit duration
  %(prog)s --dataset-root /path/to/data --out /path/to/output --skip-roughness --skip-fluctuation --max-duration-seconds 10

  # Debug: limit files, serial
  %(prog)s --dataset-root /path/to/data --out debug_output.parquet --limit 10 --n-jobs 1
        """,
    )

    # Required
    ap.add_argument("--dataset-root", required=True, help="Path to extracted DATASEC audio root")
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output path (.parquet/.csv) OR an output directory (recommended) for chunked writes. "
            "If a directory is provided, this script will write multiple part-*.parquet files and can resume."
        ),
    )

    # Performance options
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Number of parallel jobs (default: {cpu_count()} - 1 = {cpu_count() - 1})",
    )
    ap.add_argument("--batch-size", type=int, default=50, help="Files per batch for parallel processing (default: 50)")
    ap.add_argument("--chunk", type=int, default=100, help="Rows per parquet part when using directory output (default: 100)")
    ap.add_argument("--max-new", type=int, default=None, help="Process at most this many *new* files after resume")

    # Metric skipping options
    ap.add_argument(
        "--skip-roughness", action="store_true", help="Skip roughness_dw computation (expensive)"
    )
    ap.add_argument(
        "--skip-fluctuation", action="store_true", help="Skip fluctuation_strength_proxy computation (expensive)"
    )
    ap.add_argument(
        "--skip-tonality", action="store_true", help="Skip pr_ecma_st computation (moderate cost)"
    )

    # Duration limiting
    ap.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Limit audio analysis to first N seconds (None = analyze full file)",
    )

    # Other options
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    ap.add_argument("--resume", action="store_true", help="Resume by skipping ids already present in output")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging per file")

    args = ap.parse_args()

    # Validate performance settings
    if args.n_jobs < 1:
        ap.error("--n-jobs must be >= 1")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")

    # Print configuration
    print("=" * 70)
    print("DataSEC Metrics Extraction - OPTIMIZED")
    print("=" * 70)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output: {args.out}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk}")
    print(f"Max new files: {args.max_new or 'no limit'}")
    print("-" * 70)
    print("Metrics configuration:")
    print(f"  Loudness (Zwicker): ENABLED")
    print(f"  Sharpness (DIN): ENABLED")
    print(f"  Roughness: {'SKIPPED' if args.skip_roughness else 'ENABLED'}")
    print(f"  Fluctuation strength: {'SKIPPED' if args.skip_fluctuation else 'ENABLED'}")
    print(f"  Tonality (PR ECMA): {'SKIPPED' if args.skip_tonality else 'ENABLED'}")
    print(f"  Max duration: {args.max_duration_seconds or 'full file'} seconds")
    print("=" * 70)

    # Load file list
    dataset_root = Path(args.dataset_root)
    wavs = iter_wavs(dataset_root)
    print(f"Found {len(wavs)} WAV files in dataset")

    if args.limit is not None:
        wavs = wavs[: int(args.limit)]
        print(f"Limited to {len(wavs)} files for debugging")

    # Setup output
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
            print(f"Resuming: {len(done_ids)} files already completed")
    else:
        out_dir = out_target.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = None
        done_ids_path = None
        done_ids = set()

    # Provenance metadata
    provenance = {
        "dataset": "DataSEC",
        "zenodo_record": "https://zenodo.org/records/17033970",
        "dataset_root": str(dataset_root),
        "mosqito": "1.2.1",
        "note": "Metrics computed on uncalibrated waveforms; absolute values depend on SPL calibration.",
        "script_version": "v2-optimized",
        "metrics_config": {
            "skip_roughness": args.skip_roughness,
            "skip_fluctuation": args.skip_fluctuation,
            "skip_tonality": args.skip_tonality,
            "max_duration_seconds": args.max_duration_seconds,
        },
        "performance_config": {
            "n_jobs": args.n_jobs,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk,
        },
    }

    # Filter out already-completed files
    wavs_to_process = []
    for wav_path in wavs:
        rel = wav_path.relative_to(dataset_root)
        wav_id = rel.as_posix()
        if out_is_dir and args.resume and wav_id in done_ids:
            continue
        wavs_to_process.append(wav_path)

    print(f"Files to process: {len(wavs_to_process)}")

    if args.max_new is not None and len(wavs_to_process) > args.max_new:
        print(f"Limiting to {args.max_new} new files (--max-new)")
        wavs_to_process = wavs_to_process[: args.max_new]

    if not wavs_to_process:
        print("No files to process. Exiting.")
        return

    # Process in batches
    buf: list[dict] = []
    wrote = 0
    part_idx = 0
    total_start_time = time.time()

    def flush_chunk() -> None:
        nonlocal buf, wrote, part_idx
        if not buf:
            return
        df_chunk = pd.DataFrame(buf)
        if out_is_dir:
            assert parts_dir is not None
            part_path = parts_dir / f"part-{part_idx:05d}.parquet"
            df_chunk.to_parquet(part_path, index=False)
            print(f"Wrote part-{part_idx:05d}.parquet ({len(df_chunk)} rows)")
            part_idx += 1
        else:
            # For single-file output, accumulate in buf
            pass
        buf = []

    # Process batches
    n_batches = (len(wavs_to_process) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(n_batches), desc="Processing batches", unit="batch"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(wavs_to_process))
        batch_wavs = wavs_to_process[batch_start:batch_end]

        # Process batch in parallel
        batch_results = []
        if args.n_jobs == 1:
            # Serial mode (useful for debugging)
            for wav_path in batch_wavs:
                try:
                    result = compute_row(
                        wav_path,
                        dataset_root,
                        skip_roughness=args.skip_roughness,
                        skip_fluctuation=args.skip_fluctuation,
                        skip_tonality=args.skip_tonality,
                        max_duration_sec=args.max_duration_seconds,
                        verbose=args.verbose,
                    )
                    batch_results.append(result)
                except Exception as e:
                    # Create error row
                    rel = wav_path.relative_to(dataset_root)
                    parts = rel.parts
                    label = parts[0] if len(parts) >= 2 else ""
                    sublabel = parts[1] if len(parts) >= 3 else ""
                    batch_results.append(
                        {
                            "id": rel.as_posix(),
                            "audio_path": str(wav_path),
                            "text": label,
                            "label": label,
                            "sublabel": sublabel,
                            "sample_rate": None,
                            "duration_s": None,
                            "loudness": float("nan"),
                            "sharpness": float("nan"),
                            "roughness": float("nan") if not args.skip_roughness else None,
                            "fluctuation_strength_proxy": float("nan") if not args.skip_fluctuation else None,
                            "pr_ecma_st": float("nan") if not args.skip_tonality else None,
                            "error": repr(e),
                            "compute_time_s": 0.0,
                        }
                    )
                    if args.fail_fast:
                        raise
        else:
            # Parallel mode with joblib
            try:
                batch_results = Parallel(
                    n_jobs=args.n_jobs,
                    verbose=0,
                    backend="loky",
                    prefer="processes",
                )(
                    delayed(compute_row)(
                        wav_path,
                        dataset_root,
                        skip_roughness=args.skip_roughness,
                        skip_fluctuation=args.skip_fluctuation,
                        skip_tonality=args.skip_tonality,
                        max_duration_sec=args.max_duration_seconds,
                        verbose=False,  # Disable verbose in parallel to avoid clutter
                    )
                    for wav_path in batch_wavs
                )
            except Exception as batch_error:
                # If batch fails, try to process serially with error handling
                print(f"Parallel batch failed: {batch_error}, falling back to serial for this batch")
                for wav_path in batch_wavs:
                    try:
                        result = compute_row(
                            wav_path,
                            dataset_root,
                            skip_roughness=args.skip_roughness,
                            skip_fluctuation=args.skip_fluctuation,
                            skip_tonality=args.skip_tonality,
                            max_duration_sec=args.max_duration_seconds,
                            verbose=False,
                        )
                        batch_results.append(result)
                    except Exception as e:
                        rel = wav_path.relative_to(dataset_root)
                        parts = rel.parts
                        label = parts[0] if len(parts) >= 2 else ""
                        sublabel = parts[1] if len(parts) >= 3 else ""
                        batch_results.append(
                            {
                                "id": rel.as_posix(),
                                "audio_path": str(wav_path),
                                "text": label,
                                "label": label,
                                "sublabel": sublabel,
                                "sample_rate": None,
                                "duration_s": None,
                                "loudness": float("nan"),
                                "sharpness": float("nan"),
                                "roughness": float("nan") if not args.skip_roughness else None,
                                "fluctuation_strength_proxy": float("nan") if not args.skip_fluctuation else None,
                                "pr_ecma_st": float("nan") if not args.skip_tonality else None,
                                "error": repr(e),
                                "compute_time_s": 0.0,
                            }
                        )
                        if args.fail_fast:
                            raise

        # Add results to buffer
        buf.extend(batch_results)
        wrote += len(batch_results)

        # Track done IDs
        if out_is_dir:
            assert done_ids_path is not None
            with done_ids_path.open("a", encoding="utf-8") as f:
                for result in batch_results:
                    f.write(result["id"] + "\n")

        # Flush chunks
        if out_is_dir and len(buf) >= args.chunk:
            flush_chunk()

        # Print progress with ETA
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - total_start_time
            files_per_sec = wrote / elapsed if elapsed > 0 else 0
            remaining = (len(wavs_to_process) - wrote) / files_per_sec if files_per_sec > 0 else 0
            print(
                f"Progress: {wrote}/{len(wavs_to_process)} files | "
                f"Speed: {files_per_sec:.2f} files/s | "
                f"ETA: {remaining/60:.1f} min"
            )

    # Final flush
    if out_is_dir:
        flush_chunk()
        prov_path = out_dir / "provenance.json"
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print("=" * 70)
        print(f"Completed! Wrote ~{wrote} rows into {parts_dir} (chunked)")
        print(f"Provenance: {prov_path}")
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
        print("=" * 70)
        print(f"Completed! Wrote {len(df_out)} rows to {out_target}")
        print(f"Provenance: {prov_path}")

    total_elapsed = time.time() - total_start_time
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    if total_elapsed > 0:
        print(f"Average speed: {wrote/total_elapsed:.2f} files/second")
    print("=" * 70)


if __name__ == "__main__":
    main()
