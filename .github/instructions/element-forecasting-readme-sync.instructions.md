---
description: "Use when modifying OceanRace element forecasting model code, training logic, or config. Enforce README synchronization and documentation consistency checks."
name: Element Forecasting README Sync
applyTo:
  - "src/element_forecasting/**/*.py"
  - "configs/element_forecasting/*.yaml"
  - "scripts/04_train_forecast.py"
---
# Element Forecasting README Sync Rule

When this task modifies element forecasting model-related files, you must also update the module README in the same task.

## Required README Target

- Update `src/element_forecasting/README.md`.

## What Counts as Model-Related Changes

- Model architecture or forward behavior.
- Training strategy or loss composition (e.g., rollout, scheduled sampling, weighting).
- Dataset interface, feature variables, normalization assumptions, or tensor shape contract.
- Inference behavior and post-processing (e.g., overlap blend).
- Config keys/defaults in `configs/element_forecasting/model.yaml` and `configs/element_forecasting/train.yaml`.
- Entrypoint usage for forecast training in `scripts/04_train_forecast.py`.

## Required Documentation Actions

1. Map each code/config change to a README section; add a new subsection if no suitable section exists.
2. Keep names and defaults exact: parameter keys, command examples, paths, and output directories.
3. If reproducibility is affected, add migration or compatibility notes.
4. Keep edits minimal and scoped; do not rewrite unrelated sections.

## Response Requirements

When reporting completion, include:

1. A "README update mapping" table with columns:
   - change point
   - README section
   - updated content summary
   - evidence location (file + line)
2. Consistency check results (pass/fail items).

## Do Not Skip

- Do not skip README updates for "small" model-related changes.
- Do not claim "README updated" without concrete section-level details.
