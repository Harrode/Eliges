# EligES — Clinical Trial Eligibility Screening

Hybrid pipeline: **LLM parse → SemanticViewBuilder → ES two-stage retrieval → rule filter → optional selective LLM judge**.

This repository supports the **EMNLP System Demonstrations** paper and the public live demo. It is self-contained: no private paths or credentials are required to run the interactive demo.

## Architecture

```
Natural language criterion q
        │
        ▼
  L1  LLM NLU (+ silent generic defaults)  →  structured conditions c
        │
        ▼
  L2+L3  SemanticViewBuilder + two-stage ES  (N → K1=100 → K2=50)
        │
        ▼
  L4  rule_filter (labs, age, exclusions)
        │
        ▼
  L4.2  selective llm_judge on ES candidates only
        │
        ▼
  L5  optional cohort analysis / QA
```

## Quick Start (Live Demo)

**Requirements:** Python 3.10+, Docker (for Elasticsearch), an OpenAI-compatible LLM API key.

```bash
git clone <REPO_URL>
cd eliges
cp .env.example .env          # set LLM_API_KEY
./start.sh                    # ES + backend
# open http://localhost:8000
```

The demo ships **five PHI-free synthetic cohorts** (`data/synthetic/emr_*.json`, 500 patients each). Use the dataset selector in the UI to switch cohorts. No credentialed dataset is required for the booth demo.

### Manual start

```bash
pip install -r requirements.txt
python scripts/init_es.py     # optional: legacy 10-patient mock index
python backend.py
```

On first launch the backend auto-indexes synthetic cohorts into Elasticsearch when ES is available.

## Configuration

| File | Purpose |
|------|---------|
| `config/semantic_views.json` | ES query whitelist (SemanticViewBuilder) — **core system**, not a user-facing “profile pack” |
| `config/field_mapping.json` | LLM output → ES field mapping |
| `config/cohort_profiles/generic.json` | Zero-config defaults loaded silently for `emr_*` indices |
| `config/cohort_profiles/n2c2_track1.json` | Reference profile for **offline** n2c2 reproduction only |

Live demo uses the **generic** path automatically. Optional reference profiles exist for benchmark replication, not for booth interaction.

Environment variables (see `.env.example`):

| Variable | Demo | Offline eval |
|----------|------|--------------|
| `LLM_API_KEY` | required | required |
| `ES_HOST` | default `http://localhost:9200` | same |
| `N2C2_DATA_ROOT` | not needed | n2c2 2018 Track 1 root |
| `CHIA_DATA_ROOT` | not needed | Chia brat corpus |
| `N2C2_TRACK2_DATA_ROOT` | not needed | optional Track 2 scripts |

## Reproducing Paper Numbers

Obtain [n2c2 2018 Track 1](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) under its DUA, then:

```bash
export N2C2_DATA_ROOT=/path/to/n2c2_2018_track1_cohort_selection
python tests/ablation_n2c2.py              # Table 2 ablation (peak F1)
python tests/eval_hybrid.py --profile generic
python tests/eval_hybrid.py --profile n2c2_track1
```

See **`demo/baselines.md`** for headline metrics and **`REPO_UPLOAD.md`** for the full file manifest.

Chia / Track 2 evaluations additionally require `CHIA_DATA_ROOT` and `N2C2_TRACK2_DATA_ROOT` respectively.

## Key Paths

```
backend.py                      # FastAPI + 5-layer search pipeline
app/pipeline_hybrid.py          # Hybrid eval path
app/cohort_profile.py           # Profile loader (generic default)
config/                         # Semantic view + field mapping + profiles
data/synthetic/                 # PHI-free demo cohorts (included)
tests/ablation_n2c2.py          # Table 2
tests/eval_hybrid.py            # Table 5 operating points
demo/                           # Paper drafts + system docs
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [REPO_UPLOAD.md](REPO_UPLOAD.md) | GitHub release / anonymous upload checklist |
| [demo/README.md](demo/README.md) | Paper document index |
| [demo/paper_draft.md](demo/paper_draft.md) | EMNLP submission draft |
| [demo/baselines.md](demo/baselines.md) | Baseline definitions & headline F1 |
| [doc/experiments.md](doc/experiments.md) | Full experiment setup |
| [demo/system_documentation.md](demo/system_documentation.md) | System reference |

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this system in research, please cite the EMNLP 2026 System Demonstrations paper (bibtex TBD upon acceptance).
