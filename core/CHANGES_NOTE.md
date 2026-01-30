# Changes Note (baseline audit + fixes)

## Task 0 — Baseline audit (claims-as-code)

**Headline claims currently weaker than intended (from `results/paper/claims.md`):**
- Familiarity novelty-sensitivity deltas are ~0 with CIs crossing 0:
  - `fam_delta_scenic_entry_recon` = 0.02 [-0.08, 0.14]
  - `fam_delta_scenic_entry_humphrey` = 0.01 [-0.07, 0.09]
  These should be materially non-zero for non-affect variants.
- Exploratory play: ReCoN exploration is low and scanning is zero:
  - `play_unique_viewpoints_recon` = 9 (flat CI), `play_scan_events_recon` = 0.
  This suggests the recon_epistemic / recon_curiosity configs are not taking effect.
- Goal-directed corridor: `humphrey_barrett` success is low despite zero hazards (see `results/goal-directed/corridor_summary.csv`).
  This implies policy is optimizing local internal preference without a progress drive.

**Likely code paths responsible:**
- Familiarity deltas: `experiments/familiarity_control.py` uses `cw.choose_action_feelings(...)` with `w_progress=0.20` and `w_epistemic=0.0`; default scoring from agent weights dominates. Suspected precedence bug in `experiments/corridor_exp.py:choose_action_feelings` and `experiments/gridworld_exp.py:choose_action_feelings` where `agent.score_weights['w_epistemic']` overrides call-site values.
- Exploratory play (unique viewpoints + scanning): `experiments/exploratory_play.py` relies on `gw.choose_action_feelings(...)` with `w_epistemic=0.35`, but gridworld adapter does not mark bumps; forward-into-wall is not penalized (`core/driver/env_adapters.py:gridworld_adapter`). This can stall exploration and suppress scanning metrics.
- Goal-directed corridor success for HB: `experiments/evaluation_harness.rollout_episode` calls `cw.choose_action_feelings` without a progress term, while `core/driver/active_perception.py` adds an arousal-driven turn bonus. In `core/envs/corridor.py`, the “temptation” beauty bump near the hazard encourages loitering. This yields low success with zero hazards.

## Pending / to be updated
This note will be extended with the concrete fixes and their mapping to Sentience stages once changes below are implemented.

## Implemented fixes (mechanism + construct validity)

- **Task 1 — Epistemic weight precedence (core correctness)**
  - Explicit `w_epistemic` now overrides agent defaults; if omitted, agent defaults apply.
  - Code: `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (`choose_action_feelings`).
  - Test: `tests/test_epistemic_weight_precedence.py`.

- **Task 2 — Gridworld bumps are real bumps (construct validity)**
  - Forward-into-wall now sets `bumped=True` and applies a penalty.
  - Code: `core/driver/env_adapters.py` (`gridworld_adapter`), `core/envs/gridworld.py` (`bump_penalty`).
  - Test: `tests/test_gridworld_bumps.py`.

- **Task 3 — Clean stage separation (no affect nodes/state when disabled)**
  - `Ni/Nv/Na` nodes are only created when affect is enabled.
  - Affect state keys are removed when affect is disabled to avoid stale values in scoring.
  - Code: `core/ipsundrum_model.py` (`stage_C`, `stage_D`, `_attach_loop`), `core/driver/ipsundrum_dynamics.py`, `experiments/evaluation_harness.py` (read_state NaN cleanup).
  - Test: `tests/test_affect_nodes.py`.

- **Task 4 — Stage B efference copy = motor-command copy**
  - Stage B efference (`Ne`) now low-pass copies the reflexive motor command derived from `Ns`, using a filtered copy mechanism with per-tick update.
  - Code: `core/ipsundrum_model.py` (`stage_B`).
  - Test: `tests/test_stage_b_efference_copy.py` (checks filtered tracking).

- **Stage-B / non-affect: no pleasantness benefit from negative I**
  - When affect is disabled, negative sensory drive is rectified to zero before updating `N^s`.
  - Prevents ReCoN / Ipsundrum (no affect) from showing scenic preference in qualiaphilia/familiarity without Stage-D affect.
  - Code: `core/driver/ipsundrum_dynamics.py`, `core/driver/recon_forward.py`, `core/ipsundrum_model.py` (`stage_B` update).

- **Pain-tail $N^s$ half-life metric is baseline-corrected (construct validity)**
  - Fixes the strict-inequality saturation artifact when $N^s$ returns exactly to baseline (often 0.5), which previously made non-affect variants look artificially ``persistent.''
  - Code: `experiments/pain_tail_assay.py` (`baseline_corrected_half_life` + baseline rollout).
  - Test: `tests/test_pain_tail_half_life.py`.

- **Familiarity: counterfactual split-point affect probe (diagnostic robustness)**
  - Logs a horizon-5 rollout prediction for scenic-turn vs dull-turn so internal affect diagnostics are not conditioned on rare dull visits for HB.
  - Code: `experiments/familiarity_control.py` (+ `analysis/paper_claims.py` new `fam_probe_*` claims).

- **Task 5 — Corridor loitering fix (theory-consistent)**
  - Turn bonus is now gated by sensory change (uncertainty/information gain) rather than raw arousal.
  - Goal-directed corridor runs now use a default progress term from agent weights.
  - Code: `core/driver/active_perception.py` (turn bonus), `experiments/corridor_exp.py` (w_progress default), `experiments/evaluation_harness.py` (corridor `w_progress` default).

## Sentience stage mapping (a–d)

- **(a) Reflex / sentition:** Gridworld bumps + true wall penalties preserve reflexive sensorimotor consequences (`core/driver/env_adapters.py`, `core/envs/gridworld.py`).
- **(b) Efference copy:** Stage B efference now copies the outgoing motor command (reflex) rather than deriving from `Ns` directly (`core/ipsundrum_model.py`).
- **(c) Thick moment:** Ipsundrum dynamics remain unified via `ipsundrum_step`; no change to integrator mechanics beyond affect gating.
- **(d) Privatized loop / attractor:** Affect and recurrence remain isolated to stages C/D; non-affect models do not instantiate affect nodes/state.
