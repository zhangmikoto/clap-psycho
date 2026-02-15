# datasetSEC / datasheet — DataSEC (Zenodo 17033970)

## Provenance
- Zenodo record: https://zenodo.org/records/17033970
- DOI: https://doi.org/10.5281/zenodo.17033970
- Title: **DataSEC - Dataset for Sound Event Classification of environmental noise**
- License: **CC-BY-NC-SA-4.0** (`cc-by-nc-sa-4.0` in Zenodo metadata)
- Archive:
  - `DATASEC.zip` (md5: `29fa9b8cc84cfa69aa4e5e674e780383`)

## Local layout (this machine)
- Download dir: `/root/clawd/_datasets/zenodo_17033970/`
- Zip: `/root/clawd/_datasets/zenodo_17033970/DATASEC.zip`
- Extracted audio root (expected): `/root/clawd/_datasets/zenodo_17033970/DATASEC/DATASEC/`

Folder structure appears label-based:
- `<dataset-root>/<label>/*.wav`
- and sometimes nested:
  - `<dataset-root>/<label>/<sublabel>/*.wav`

## Dataset composition (quick inspection)
- Audio format: WAV
- Observed in header scan (files that `wave` can read):
  - sample rate: 44.1 kHz
  - channels: mono
  - sample width: 16-bit (2 bytes)
- Duration stats (header-derived):
  - min: ~0.20 s
  - median: ~8.25 s
  - mean: ~16.66 s
  - p95: ~48.82 s
  - max: ~481.48 s
  - total: ~22.73 hours
- Classes: 20 top-level labels (examples):
  - `Vehicle pass-by`, `Vehicle idling`, `Voices`, `Music`, `Horn`, `Sirens and alarms`, `Train`, `Jet aircrafts`, `Propeller aircrafts`, `Bells`, `Glass breaking`, ...

## Metrics pipeline
### Goal
Compute psychoacoustic **building-block** metrics per clip (features suitable for downstream ML).

### Resampling / calibration policy
- MOSQITO can emit warnings about resampling to 48 kHz depending on the metric implementation.
- **Calibration note:** MOSQITO expects signals in Pa for absolute SPL. These waveforms are uncalibrated; absolute values are not physically meaningful, but relative values can still be useful for learning.

### Implementation
Script:
- `scripts/zenodo_17033970_compute_metrics.py`

Recommended durable output layout:
- Output dir: `/root/clawd/_datasets/zenodo_17033970/metrics/`
  - `done_ids.txt` (one id per processed file; used for resume)
  - `parts/part-*.parquet` (chunked outputs)
  - `provenance.json`

Command (recommended):
```bash
cd /root/clawd/clap-psycho
. .venv/bin/activate
python scripts/zenodo_17033970_compute_metrics.py \
  --dataset-root /root/clawd/_datasets/zenodo_17033970/DATASEC/DATASEC \
  --out /root/clawd/_datasets/zenodo_17033970/metrics \
  --chunk 100 \
  --resume
```
Batch mode example (safer for long runs):
```bash
python scripts/zenodo_17033970_compute_metrics.py \
  --dataset-root /root/clawd/_datasets/zenodo_17033970/DATASEC/DATASEC \
  --out /root/clawd/_datasets/zenodo_17033970/metrics \
  --chunk 100 \
  --resume \
  --max-new 200
```

## Output schema (parquet columns)
Per row (one audio file):
- Identifiers:
  - `id` (string; relative path under dataset-root)
  - `audio_path` (string; absolute path)
- Text/labels:
  - `text` (string; currently uses `label` as placeholder)
  - `label` (string; top-level folder)
  - `sublabel` (string; second-level folder if present)
- Audio info:
  - `sample_rate` (int)
  - `duration_s` (float)
- Psychoacoustic building blocks:
  - `loudness` (float; Zwicker stationary)
  - `sharpness` (float; DIN)
  - `roughness` (float; Daniel & Weber; max over time)
  - `fluctuation_strength_proxy` (float; proxy via mosqito loudness_zwtv; very expensive; often skipped)
  - `pr_ecma_st` (float; prominence ratio / tonality building block)
- Cheap temporal-modulation substitutes (computed from Hilbert envelope, 0.5–20 Hz bandpass):
  - `fluctuation_evr` (float; envelope variance ratio)
  - `fluctuation_ami` (float; amplitude modulation index)
  - `fluctuation_rms_mod` (float; RMS of modulation-band envelope)
- Error handling:
  - `error` (string; empty if OK)

## Reproducibility notes
- Keep `DATASEC.zip` out of git.
- Suggested: record `pip freeze` and git commit hash of `clap-psycho` when running the extraction.
