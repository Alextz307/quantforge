/**
 * Pre-fill content for the "New spec" Monaco editor.
 *
 * A fully-commented one-leg StudySpec covering every required + optional
 * field. The user edits in place rather than authoring from scratch — every
 * required field is present, every optional field shows its default with an
 * explanatory comment, and constraints documented on each Pydantic
 * ``description=`` are echoed in inline comments so the editor surface mirrors
 * the schema even before autocomplete fires.
 *
 * Keep this in sync with the descriptions on ``StudySpec`` / ``StudyLeg`` in
 * ``src/core/config.py``. The annotated example is the single most important
 * piece of YAML-format clarity in the whole flow — drift here means worse
 * first-use UX.
 */
export const STUDY_SPEC_SKELETON = `# Study spec — sweeps one or more strategies across multiple universes,
# tunes each (strategy, universe) leg via HPO, then optionally runs a
# holdout eval. See "Format help" on the right for the field reference.
name: my_first_study                  # required — slug used for artifact dirs
description: >                        # optional — surfaced in the webapp
  Replace this with what you are testing and why.
seed: 42                              # optional, default 42 — Optuna + numpy seed
output_dir: studies/my_first_study    # required — artifact root under store_root

# legs[]: one entry per strategy. Each leg sweeps the strategy across its
# universe list. Strategy names must be unique across legs.
legs:
  - strategy: AdaptiveBollinger       # required — must match a registered strategy
    strategy_config: config/strategies/adaptive_bollinger.yaml  # required path
    hpo_config: config/hpo/adaptive_bollinger.yaml              # required path
    universes:                        # required — at least one, no duplicates
      - spy_daily_5y
      - qqq_daily_5y
`;

/**
 * 4-line skeleton pasted by the "Insert leg" quick action in the help panel.
 *
 * Indentation matches the standard 2-space block style under ``legs:`` — the
 * paste lands cleanly when inserted at the start of an existing list entry's
 * line. The trailing newline keeps cursor positioning predictable.
 */
export const STUDY_LEG_SKELETON = `  - strategy: ReplaceMe
    strategy_config: config/strategies/replace_me.yaml
    hpo_config: config/hpo/replace_me.yaml
    universes:
      - spy_daily_5y
`;
