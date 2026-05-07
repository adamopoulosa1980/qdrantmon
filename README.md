# Qdrant Health Monitor

A Python tool that inspects Qdrant collections and produces a health report. It auto-detects whether each collection uses dense, sparse, or named vectors and runs the metrics appropriate to that type.

## Features

- Auto-detects vector type per collection (`dense`, `sparse`, `named`)
- Per-type quality metrics (magnitude, sparsity, vocabulary coverage, etc.)
- Duplicate / near-duplicate detection on a sampled subset
- PCA-based semantic diversity score
- Pairwise-distance drift score (within a single run, against the first sample seen)
- Composite health score (0–100) with rule-based deductions
- Console output and Markdown report export
- Configuration via `.env`, with CLI flag overrides

## Installation

```bash
pip install -r requirements_qdrant_monitor.txt
```

Dependencies (see [requirements_qdrant_monitor.txt](requirements_qdrant_monitor.txt)):

- `qdrant-client>=1.15.0`
- `numpy>=1.24.0`
- `scipy>=1.11.0`
- `scikit-learn>=1.3.0`
- `python-dotenv>=1.0.0`
- `prometheus-client>=0.19.0`

## Configuration

Copy [.env.example](.env.example) to `.env` and edit:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
SAMPLE_SIZE=1000
SIMILARITY_THRESHOLD=0.99
OUTPUT_DIR=./health_reports
LOG_LEVEL=INFO
```

| Variable | Used for | Default |
| --- | --- | --- |
| `QDRANT_URL` | Qdrant server endpoint | `http://localhost:6333` |
| `QDRANT_API_KEY` | API key (Qdrant Cloud / auth) | unset |
| `SAMPLE_SIZE` | Points sampled per collection | `1000` |
| `SIMILARITY_THRESHOLD` | Cosine threshold for near-duplicates | `0.99` |
| `OUTPUT_DIR` | Directory for Markdown reports | `./health_reports` |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

## Usage

### CLI

```bash
# All collections, console output only
python qdrantmon.py

# All collections, write Markdown report to OUTPUT_DIR
python qdrantmon.py --markdown

# Single collection
python qdrantmon.py --collection my_collection --markdown

# Override .env settings
python qdrantmon.py --url http://qdrant.example.com:6333 --api-key SECRET --markdown
python qdrantmon.py --markdown --output-dir ./custom_reports
```

| Flag | Effect |
| --- | --- |
| `--url` | Override `QDRANT_URL` |
| `--api-key` | Override `QDRANT_API_KEY` |
| `--collection NAME` | Analyze a single collection (default: all) |
| `--markdown` | Write a Markdown report and exit (otherwise prints to console) |
| `--output-dir DIR` | Override `OUTPUT_DIR` (only meaningful with `--markdown`) |

When `--markdown` is set the script writes `qdrant_health_report_YYYYMMDD_HHMMSS.md` under `OUTPUT_DIR` and exits without printing the console summary.

### Python API

```python
from qdrantmon import QdrantHealthMonitorEnhanced

monitor = QdrantHealthMonitorEnhanced()              # uses .env
report = monitor.generate_report()                    # all collections
report = monitor.generate_report(["my_collection"])   # subset

# Lower-level entry points
vinfo  = monitor.detect_vector_type("my_collection")
dense  = monitor.analyze_dense_vectors("my_collection")
sparse = monitor.analyze_sparse_vectors("my_collection")
named  = monitor.analyze_named_vectors("my_collection")
health = monitor.compute_collection_health("my_collection")
```

To produce a Markdown file programmatically:

```python
from qdrantmon import QdrantHealthMonitorEnhanced
from markdown_report_generator_enhanced import MarkdownReportGeneratorEnhanced

monitor = QdrantHealthMonitorEnhanced()
generator = MarkdownReportGeneratorEnhanced(monitor)
path = generator.generate_markdown_report(output_dir="./health_reports")
```

## Vector type analysis

### Dense vectors

Implemented in [qdrantmon.py:196](qdrantmon.py#L196). Computes:

- `avg_magnitude`, `std_magnitude`, `min_magnitude`, `max_magnitude` — L2 norms across the sample
- `nan_count` — total NaN entries across the matrix
- `zero_count` — vectors that are all-zero
- `duplicate_pairs`, `near_duplicate_pairs` — exact (cosine > 0.9999) and near (cosine > `SIMILARITY_THRESHOLD`) pair counts on a sub-sample of up to 500 vectors, scaled to the full sample size
- `approximate_diversity` — sum of explained-variance ratios from a PCA with up to 10 components on standardized vectors

### Sparse vectors

Implemented in [qdrantmon.py:273](qdrantmon.py#L273). Computes:

- `avg_non_zero_count`, `min_non_zero_count`, `max_non_zero_count`
- `avg_sparsity` (%) — fraction of zero dimensions, estimated from observed indices
- `coverage_score` (%) — fraction of the inferred vocabulary actually used
- `sparsity_pattern` — bucketed label: `extreme` (>99.9%), `very_high` (99–99.9%), `high` (95–99%), `moderate` (80–95%), `low` (<80%)

### Named vectors

Implemented in [qdrantmon.py:355](qdrantmon.py#L355). For each named vector, computes magnitude stats (`avg`, `std`, `min`, `max`), `nan_count`, and PCA-based `diversity`.

### Drift

[qdrantmon.py:567](qdrantmon.py#L567) computes mean pairwise cosine distance and clustering tightness on the current sample. The first call per collection becomes the in-process reference; subsequent calls return a `drift_score` relative to it. **The reference is not persisted across runs** — drift is only meaningful within a single Python process.

## Health scoring

Implemented in [qdrantmon.py:412](qdrantmon.py#L412). Each collection starts at 100 and is deducted as follows:

**Dense:**

| Condition | Deduction |
| --- | --- |
| `avg_magnitude` outside [0.8, 1.2] | -15 |
| any NaN vectors | -25 |
| any all-zero vectors | -15 |
| `approximate_diversity < 0.3` | -12 |

**Sparse:**

| Condition | Deduction |
| --- | --- |
| `coverage_score < 0.1` (%) | -20 |
| `sparsity_pattern == 'extreme'` | -10 |
| `avg_non_zero_count < 3` | -15 |

**Named (per named vector):**

| Condition | Deduction |
| --- | --- |
| any NaN values | -10 |
| `avg_magnitude` outside [0.8, 1.2] | -8 |

**All types:** drift score > 0.2 → -15.

Score is clamped to [0, 100].

## Markdown report

Generated by [markdown_report_generator_enhanced.py](markdown_report_generator_enhanced.py). Sections:

1. **Header** — timestamp, Qdrant URL, collection count, average health
2. **Executive Summary** — overall status, top critical issues, per-collection table with score bars
3. **Vector Types Summary** — counts per type and a comparison table
4. **Collection Analysis** — per-collection metric tables driven by the detected vector type
5. **Metric Reference Guide** — what each number means and its target range
6. **How to Use This Report** — interpretation guidance

Filename pattern: `qdrant_health_report_YYYYMMDD_HHMMSS.md`.

## Project layout

| File | Role |
| --- | --- |
| [qdrantmon.py](qdrantmon.py) | Main monitor: vector-type detection, per-type analysis, health scoring, CLI |
| [markdown_report_generator_enhanced.py](markdown_report_generator_enhanced.py) | Markdown report writer used when `--markdown` is set |
| [requirements_qdrant_monitor.txt](requirements_qdrant_monitor.txt) | Python dependencies |
| [.env.example](.env.example) | Configuration template |
| [health_reports/](health_reports/) | Default output directory for Markdown reports |
