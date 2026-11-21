# A5P5 code - GRU sequence classifier for crash-consistent credential rotation

Run (from this folder):

    pip install -r requirements.txt   # plus the Graphviz `dot` binary
    python3 run_all.py

The script reads `../data/A5P5_credential_rotations.csv`, uses fixed seeds (2026 + fold) and single-threaded
PyTorch, and runs in roughly 6 minutes on 2 CPU cores.

What it does
- Fold-wise preprocessing: median/mode imputation with missing-value flags, log transform and P1-P99
  winsorisation of replication lag and transaction rate. Post-event columns (final_state, auth_failure_minutes,
  failed_transactions, missed_blocks) are never used as features; final_state is only an auxiliary training label.
- Encodes each rotation run as a 6-step event sequence (CREATE, STORE_WRITE, TARGET_UPDATE, CLIENT_RELOAD,
  REVOKE_OLD, RECOVER) with step status (completed / crashed here / not reached / recovery / no-op), an
  "interrupted mid-step" flag at CLIENT_RELOAD (separates after_db_or_node_update from during_client_reload), plus a
  static context vector.
- Proposed GRU-CR: context-initialised GRU + binary risk head + auxiliary final-state head.
- Baselines: fixed-order rotation rules (expert checklist score), logistic regression, MLP (scikit-learn).
- Stratified 5-fold CV (inner 15% validation for early stopping and F1-optimal threshold), ablation study,
  hidden-size sensitivity, paired t-tests, out-of-fold per-crash-point analysis and a protocol counterfactual map.

Outputs (../results unless noted)
- table1_dataset_summary.csv, table3_main_comparison.csv, table4_ablation.csv, sensitivity_hidden_summary.csv
- cv_fold_metrics.csv, ablation_fold_metrics.csv, sensitivity_hidden_fold_metrics.csv (per-fold values)
- gru_training_log.csv, per_crash_point_pr_auc.csv, protocol_crash_inconsistency_rate.csv,
  counterfactual_protocol_risk_gru.csv, metrics.json (all headline numbers, t-tests, runtime, and the
  generator-oracle reference: the data-generating equation of gen_author5.py applied to the observed columns,
  with the latent pinned-client flag and N(0,0.55) noise marginalised)
- fig1..fig7 as JPG here and as 300-dpi PNG in ../figures (fig1 = Graphviz methodology flowchart)
