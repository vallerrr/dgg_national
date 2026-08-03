

The canonical rules for this project — the stable standards every notebook and script follows. This file holds **rules**; the individual **records** they govern (data decisions, methodology narrative) live in `doc/`. Copy this file into a new project and adjust the bracketed values.

## 1. File structure

```
project/
├── README.md            project overview, setup, run instructions
├── CONVENTIONS.md       this file
├── data/
│   ├── raw/                 source data — immutable, never edited in place
│   ├── processed/           derived data; reproducible from raw/
│   └── README.md            data provenance and dictionary
├── src/                 notebooks, params.py, utils.py
├── doc/
│   ├── methodology.md       methods narrative, assumptions, limitations
│   └── decisions/           data & method decision records (see §8)
├── outputs/             generated figures, tables, reports
├── logs/                machine-generated run logs
└── environment.yml
```

Paths are referenced relative to the project root via `params.py` — never hardcode absolute paths, so the project stays portable. (However, if you find the project already have file structures, please ask to re-set or not)

## 2. Documentation requirements

Document what is _lost and expensive to reconstruct_ — provenance and decisions — not structure the code already expresses. Required docs (and nothing per-folder beyond these):

- **`README.md`** (root) — what the project is, how to set up the environment, how to run the notebooks in order, and pointers to the docs below.
- **`data/README.md`** — data provenance and dictionary.
- **`doc/methodology.md`** — methods narrative, assumptions, limitations.
- **`doc/decisions/`** — the running decision records (§8).

Generated directories (`outputs/`, `logs/`) get no README — their contents are dated and machine-produced, so a static description only rots.

## 3. File naming and dates

Every generated file carries its **save date** in sortable ISO form (`YYYYMMDD`), e.g. `summary_20260704.csv`. Sortability is why ISO, not any other format.

- The date is when the file was produced, not the data's date.
- Raw data keeps its acquisition date and is never re-dated or overwritten.
- Logs use a full timestamp (`YYYYMMDD_HHMMSS`).

## 4. `params.py` and `utils.py`

Both always exist in `src/`.

- **`params.py`** — all settings: paths (relative to root), constants, config, random seeds, and the `STYLE` dict (§6). Anything a notebook would otherwise hardcode lives here.
- **`utils.py`** — shared reusable functions; notebooks import rather than redefine.

Secrets (API keys, credentials) do **not** go in `params.py`, which is committed — use environment variables or a gitignored config file.

## 5. Environment

Managed with miniforge3; the environment name is close to the project name.

```bash
mamba env list                      # find it
mamba activate <env-name>           # activate it
mamba env create -f environment.yml # or rebuild from scratch
```

Keep `environment.yml` current so the environment is reproducible, not only discoverable.

## 6. Visualization style

Every figure reads its styling from `params.STYLE` — never hardcode a colour, size, or font inline, so a single edit restyles the whole project.

```python
STYLE = {
    "colors":   ["<hex1>", "<hex2>", "<hex3>", "<hex4>"],  # project palette
    "figsize":  (12, 7),
    "title_fs": 16,
    "label_fs": 13,
    "tick_fs":  11,
    "dpi":      300,
    "save":     False,          # True → write figures to savedir
    "savedir":  "outputs/fig",
}
```

## 7. Modularization

Functions are small and single-purpose — one job, inputs as arguments, outputs returned, no reliance on notebook-global state. Develop inline while iterating; move stable functions to `utils.py` once they settle.

## 8. Decision logging (data and methodology)

Separate **rules** (this section — how decisions are made and recorded) from **records** (the decisions themselves, in `doc/decisions/`).

**No silent decisions.** Every data or methodological decision follows:

1. **State the assumption** explicitly.
2. **Fact-check against the data** — show the effect: rows before/after, records affected and %, a sample of what changed. Confirm assumptions against the data, never on faith.
3. **Record it** (see below).
4. **Escalate** substantive or uncertain calls for a human decision before proceeding — ask, don't assume.

**Where records live.** In `doc/decisions/` — curated documentation, never in `logs/` (which holds machine-generated run output).

**How records are organised — partition by scope, tag by kind:**

- _Scope sets the file:_
    - `doc/decisions/shared.md` — decisions that apply project-wide (ingestion, global standardisation, canonical dataset construction). Made once, reused.
    - `doc/decisions/<analysis_name>.md` — decisions local to one analysis. The same dataset may be handled differently across analyses; each analysis's choices live in its own file.
- _Kind is a tag on each entry_ (`data` or `method`), not a separate file — so data or method decisions can be filtered across files without the kind axis fighting the scope axis.

**Start simple, scale up.** A small project needs only a single `doc/decisions.md`. Split into the `doc/decisions/` folder (shared + per-analysis) only once analyses genuinely diverge — don't build the hierarchy prematurely.

**Methodology narrative.** `doc/methodology.md` is the readable synthesis of the `method`-kind decisions plus assumptions and limitations, for write-up. Method decisions are still logged like any other; the narrative reads from them.

## 9. Reproducibility

Any step involving randomness — sampling, splits, network layout, shuffles — sets an explicit seed recorded in `params.py`, so results are reproducible run to run.