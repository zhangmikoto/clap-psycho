# VIBE_CODING.md — clap-psycho

> A scratchpad for "vibe coding" decisions: what we’re building, why it matters, and how we want to work.

## The idea (problem statement)
Modern audio-language models (ALMs) are typically trained on large-scale **text–audio pairs** designed for categorization and semantic description.

Example reference: CLAP / LAION-Audio-630K (contrastive language–audio pretraining):
- Paper: https://arxiv.org/abs/2211.06687

These datasets are great for *"what is this sound?"* (labels, events, sources, scenes), but they largely lack **perceptual / psychoacoustic information**.

Result: even state-of-the-art ALMs (e.g., Audio Flamingo 3, Qwen3 Omni, etc.) can be weak at *"how does it feel/sound perceptually?"* questions:
- harsh vs. warm
- rough vs. smooth
- sharp vs. dull
- annoying vs. pleasant
- "more comfortable" vs. "more fatiguing"

In other words: they learn semantic/audio-event alignment, but not necessarily the **psychoacoustic perceptual space** humans care about.

## Hypothesis
If we supply training/evaluation data that explicitly encodes psychoacoustic perceptual attributes (either as labels, scalar metrics, pairwise preferences, or natural-language descriptions grounded in psychoacoustics), ALMs can learn representations that better reflect human auditory perception.

## What we mean by “psychoacoustic information”
Potential targets (examples):
- loudness (Zwicker / ECMA)
- sharpness (DIN)
- roughness (Daniel & Weber)
- fluctuation strength (often from loudness modulation, depending on implementation)
- tonality, annoyance, etc. (future)

Possible supervision forms:
- **Scalar regression**: (audio → metric value)
- **Pairwise ranking**: (audio A more rough than audio B)
- **Multi-attribute classification**: (low/med/high sharpness)
- **Text augmentation**: captions that include perceptual descriptors *grounded by metrics*

## Design principles
- **Reproducible:** metrics pipeline must be deterministic and versioned.
- **Traceable:** every label/metric should be reproducible from raw audio + config.
- **Modular:** metrics, transforms, and dataset logic should be swappable.
- **Honest about proxies:** if a metric is approximated (e.g., fluctuation strength proxy), we mark it clearly.

## Immediate next steps (TBD)
- Define a minimal dataset format (audio + metadata + psychoacoustic targets).
- Decide supervision strategy (regression vs ranking vs caption augmentation).
- Build a reference pipeline to compute targets from audio.
- Build baselines to test whether CLAP-like embeddings correlate with psychoacoustic targets.

## Working conventions
- Keep notes and rationale here.
- Keep code minimal at first: a tiny, correct pipeline beats a big messy one.
- Prefer explicit configs (YAML/JSON) for dataset generation.
