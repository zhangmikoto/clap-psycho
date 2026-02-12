# TODO: Zenodo 17033970 datasetSEC + metrics pipeline

Link: https://zenodo.org/records/17033970

## Goal
After UrbanSound8K finishes, move on to the next dataset (Zenodo record 17033970). We want to:
1) download the dataset to `_datasets/` (not git)
2) inspect file structure + metadata (captions/labels if present)
3) create a robust metrics extraction script (mosqito-based) with durable resume
4) produce a datasetSEC/datasheet describing dataset provenance + metrics schema

## Where to put things
- Download/extract: `/root/clawd/_datasets/zenodo_17033970/` (or rename once we learn the dataset name)
- Outputs: `/root/clawd/_datasets/zenodo_17033970/metrics/`
  - `done_ids.txt`
  - `parts/part-*.parquet`
- Code: `/root/clawd/clap-psycho/scripts/`
- Docs: `/root/clawd/clap-psycho/docs/` (create if missing)

## Intended agent prompt (for GLM-4.7 sub-agent)

You are working in `/root/clawd/clap-psycho`.

Task: Prepare the next dataset from Zenodo record https://zenodo.org/records/17033970.

Deliverables:
A) **Dataset discovery**
- Identify dataset name, description, license, size, and list of files from Zenodo metadata.
- Record what is audio vs metadata.

B) **Download + layout**
- Download all required files (or the smallest subset if multiple archives) into `/root/clawd/_datasets/zenodo_17033970/`.
- Use resumable downloads (aria2c -c or wget -c) and write the exact commands into a README.
- Extract archives if needed and document final folder structure.

C) **Metadata parsing**
- Determine how to iterate audio files.
- If there is metadata (CSV/JSON/JSONL/parquet), locate the columns/fields that map audio→text (caption/label) and a stable ID.
- If there is no text, propose a placeholder `text` field (e.g., filename or class label).

D) **Metrics script**
- Create a new script `scripts/zenodo_17033970_compute_metrics.py` (or better name after discovery).
- Follow the UrbanSound8K extractor pattern:
  - args: `--dataset-root`, `--out`, `--chunk`, `--resume`, `--max-new`
  - writes chunked parquet parts into `out/parts/part-00000.parquet` etc.
  - maintains `out/done_ids.txt` for resume
  - per-row include at least: `id`, `audio_path`, `text` (if available), `sr`, `duration_s`, plus psychoacoustic metrics (same set used previously), and `error` string.
  - tolerate failures: on exception, write a row with `error` populated.
  - note expected mosqito resample-to-48k warnings.

E) **datasetSEC / datasheet**
- Create `docs/datasetsec_zenodo_17033970.md` describing:
  - provenance: Zenodo URL, citation, license
  - dataset composition: number of clips, formats, sample rate distribution (if quick to compute), labels/captions availability
  - metrics: definition, units, aggregation strategy (e.g., roughness max), versions (mosqito version), resampling policy (48 kHz)
  - output schema (parquet columns)
  - reproducibility: command lines, config hash suggestion

Constraints:
- Do NOT put large data into git.
- Add/update `.gitignore` to exclude `_datasets/` and downloaded archives.
- Keep outputs durable for restarts.

Return:
- paths created
- exact next command(s) to run
- any open questions/blockers.

## Notes
- Main agent stays on OpenAI; sub-agents default to `zai/glm-4.7`.
- Container may be rebooted; assume only disk state persists.
