"""Compute psychoacoustic building-block metrics for ESC-50.

Input:
- ESC-50 repo checkout (downloaded zip)
  expects:
    <esc_root>/meta/esc50.csv
    <esc_root>/audio/*.wav

Output:
- parquet or csv with one row per audio file and associated text label.

We record *building blocks* (not a single annoyance score):
- loudness (Zwicker stationary)
- sharpness (DIN)
- roughness (Daniel & Weber)
- fluctuation strength (proxy from time-varying loudness modulation)
- tonality building block: prominence ratio (ECMA 418-1)

Notes
-----
- MOSQITO expects signal in Pa for absolute calibration; ESC-50 is arbitrary.
  We still compute metrics for relative learning, but we must document that
  absolute values depend on calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# MOSQITO
from mosqito.sq_metrics import loudness_zwst, sharpness_din_st, roughness_dw, loudness_zwtv

# tonality API (ECMA 418-1)
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
    """Proxy FS from loudness time series (0.5–20 Hz RMS band)."""
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


def tonality_scalars(x: np.ndarray, sr: int) -> dict[str, float]:
    """Compute tonality-related scalars.

    We keep a single tonality building block here:
    - prominence ratio (PR, ECMA 418-1)

    (TNR is closely related and can be added back if needed.)

    Returns
    -------
    dict with key: pr_ecma_st
    """
    pr = pr_ecma_st(x, sr)

    def to_float(v):
        """Convert MOSQITO outputs to a scalar.

        Many MOSQITO functions return either:
        - a float/np scalar
        - an ndarray
        - a tuple where the first item is the metric and the rest are details
        """
        if isinstance(v, tuple):
            v = v[0]
        if isinstance(v, np.ndarray):
            return float(np.max(v)) if v.size else 0.0
        return float(v)

    return {
        "pr_ecma_st": to_float(pr),
    }


def compute_row(wav_path: Path) -> dict:
    sr, x = load_wav(wav_path)

    # MOSQITO building blocks
    loud = float(loudness_zwst(x, sr)[0])
    sharp = float(sharpness_din_st(x, sr))

    r = roughness_dw(x, sr, overlap=0)[0]
    rough = float(np.max(r)) if isinstance(r, np.ndarray) else float(r)

    fs_proxy = fluctuation_strength_proxy(x, sr)
    ton = tonality_scalars(x, sr)

    out = {
        "sample_rate": sr,
        "duration_s": float(len(x) / sr),
        "loudness": loud,
        "sharpness": sharp,
        "roughness": rough,
        "fluctuation_strength_proxy": fs_proxy,
    }
    out.update(ton)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc-root", required=True, help="Path to ESC-50 root (contains meta/ and audio/)")
    ap.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = ap.parse_args()

    esc_root = Path(args.esc_root)
    meta_csv = esc_root / "meta" / "esc50.csv"
    audio_dir = esc_root / "audio"

    df_meta = pd.read_csv(meta_csv)
    if args.limit is not None:
        df_meta = df_meta.head(int(args.limit)).copy()

    rows = []
    for _, r in tqdm(df_meta.iterrows(), total=len(df_meta), desc="ESC-50 metrics"):
        fn = r["filename"]
        wav_path = audio_dir / fn
        base = {
            "id": fn,
            "audio_path": str(wav_path),
            # text side of the pair: use category label
            "text": str(r["category"]),
            "category": str(r["category"]),
            "target": int(r["target"]),
            "fold": int(r["fold"]),
        }

        try:
            metrics = compute_row(wav_path)
            base.update(metrics)
        except Exception as e:
            base["error"] = repr(e)
            if args.fail_fast:
                raise

        rows.append(base)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(rows)

    # Attach provenance
    df_out.attrs["provenance"] = {
        "dataset": "ESC-50",
        "esc_root": str(esc_root),
        "mosqito": "1.2.1",
        "note": "Metrics computed on uncalibrated waveforms; absolute values depend on SPL calibration.",
    }

    if out_path.suffix.lower() == ".parquet":
        df_out.to_parquet(out_path, index=False)
        # Save provenance alongside (parquet metadata handling varies)
        prov_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
        prov_path.write_text(json.dumps(df_out.attrs["provenance"], indent=2), encoding="utf-8")
    elif out_path.suffix.lower() == ".csv":
        df_out.to_csv(out_path, index=False)
        prov_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
        prov_path.write_text(json.dumps(df_out.attrs["provenance"], indent=2), encoding="utf-8")
    else:
        raise ValueError("--out must end with .parquet or .csv")

    print(f"Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
