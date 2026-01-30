# ENGINEERING_NOTES

## Humphrey stage mapping (mechanism → code)

- **Stage B — efference copy (informative, persistent)**
  - Replaced constant Ne with reflex-driven, decaying efference copy: `core/ipsundrum_model.py` (Builder.stage_B → `efference_effect`).
  - Stage-B / non-affect input uses rectified drive (no "pleasantness" from negative I): `core/ipsundrum_model.py`, `core/driver/recon_forward.py`, `core/driver/ipsundrum_dynamics.py`.
  - Exposed Ne for diagnostics: `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (agent `_read_state` + logs).
  - ReCoN baseline removes qualia-linked scoring terms and epistemic drive to keep Stage B reflexive: `experiments/evaluation_harness.py`, `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (recon `score_weights`, incl. `w_epistemic=0`).
  - Epistemic-weight precedence fixed so assay overrides are honored: `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (`choose_action_feelings` uses explicit `w_epistemic` when provided).

- **Stage C/D — privatization gating (policy-level)**
  - Internal-mode gating uses efference threshold from P to scale epistemic/curiosity and add arousal-driven scan bonus: `core/driver/active_perception.py`.
  - Policy now receives efference threshold from the running network: `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (PolicyContext).
  - Novelty is suppressed by high valence in affect-enabled models to stabilize scenic preference while preserving internal scanning via arousal bonus: `core/driver/active_perception.py`.
  - Turn incentives now depend on arousal * sensory change (information gain proxy), and caution weight is increased to curb reckless forward motion under affect: `core/driver/active_perception.py`.
  - Ipsundrum-only (no affect) models suppress epistemic drive when curiosity is off to avoid non-affective scanning from masquerading as caution: `core/driver/active_perception.py` (`w_epistemic_eff` gating with `efference_threshold` + `curiosity`).
  - Goal-directed corridor runs default a modest progress term (prevents ipsundrum loitering without per-assay hacks): `experiments/evaluation_harness.py` (`w_progress`), `experiments/corridor_exp.py` (optional `w_progress` override).

- **Stage C/D — thick moment loop (integrator fidelity)**
  - Single-source ipsundrum dynamics implemented in `core/driver/ipsundrum_dynamics.py` and used for both online update and forward rollouts:
    - Online: `core/ipsundrum_model.py` (`_attach_loop.update_sensor` → `ipsundrum_step`).
    - Forward: `core/driver/ipsundrum_forward.py` → `ipsundrum_step`.
  - Affect nodes are instantiated only when affect is enabled (no valence/arousal contamination in non-affect stages): `core/ipsundrum_model.py`, `experiments/evaluation_harness.py` (`mean_valence/mean_arousal` as NaN when absent).
  - Pain-tail assay removes the external hazard after the contact and recomputes smell to isolate internal persistence (stimulus removed): `experiments/pain_tail_assay.py`.
  - Pain-tail $N^s$ half-life metric is baseline-corrected (avoids strict-inequality saturation at $N^s\\approx0.5$): `experiments/pain_tail_assay.py` (`baseline_corrected_half_life` + baseline rollout).

- **Stage D — attractor alignment (online vs forward)**
  - Forward model now matches online loop exactly (precision/g_eff, integrator, efference, Barrett update), ensuring attractor/alpha dynamics are consistent: `core/driver/ipsundrum_dynamics.py` + `core/ipsundrum_model.py`.

## Model-variant rollout alignment

- Recon planning uses recon forward dynamics (never ipsundrum forward):
  - `experiments/gridworld_exp.py`, `experiments/corridor_exp.py` (`select_forward_model`).
  - `experiments/familiarity_control.py`, `experiments/familiarity_internal.py` (policy + forward-model selection).
  - Recon rollouts explicitly route to `core/driver/recon_forward.py` and unit test spies on `ipsundrum_forward` to prevent regressions.

## Claims + assays

- Internal actuator mode is logged and claimed from exploratory play:
  - Metric: `experiments/exploratory_play.py` (`internal_actuator_fraction`).
  - Claims: `analysis/paper_claims.py` (`play_internal_actuator_*`).
  - Gridworld bump detection now triggers penalties when forward collides with a wall, preventing boundary-stuck artifacts in play metrics: `core/driver/env_adapters.py`, `core/envs/gridworld.py`.
  - Pain-tail assay seeds increased to 20 and goal-directed horizons expanded (up to 20) to tighten CIs: `run_experiments.sh` (paper profile defaults), `experiments/goal_directed_sweeps.py` (horizons list).
  - Familiarity-control now logs a split-point counterfactual affect probe (horizon-5 rollout for scenic-turn vs dull-turn) so dull-lane affect diagnostics are not based on rare visits: `experiments/familiarity_control.py`, claims `fam_probe_*` in `analysis/paper_claims.py`.

## Results snapshot (paper profile)

- Lingering planned caution now cleanly dissociates: `pain_tail_duration_humphrey=5 [5,5]`, `pain_tail_duration_hb=90 [52,128]`, and `results/paper/dissociation_table.tex` marks HB with a checkmark.
- Baseline-corrected pain-tail $N^s$ half-life is immediate for all variants in this parameterization (`pain_ns_half_life_* = 0`), so the pain-tail plot reports $N^s$ AUC above baseline (and persistence is causally supported by lesion AUC-drop: `lesion_auc_drop_humphrey≈19.1`, `lesion_auc_drop_hb≈27.6`).
- Structured local scanning remains HB-only: `play_scan_events_hb` >> `play_scan_events_recon/humphrey`.
- Valence-stable scenic preference remains HB-only: `fam_delta_scenic_entry_hb` within ±0.05 while recon/humphrey deltas are larger.
