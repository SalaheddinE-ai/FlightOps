# Data & model lineage

Every registered version can be traced through:

`Dataset version -> Validation report -> Preprocessing config -> Feature-engineering version -> Training config -> Experiment -> Run -> Model artifact -> Registered version -> Deployment -> Prediction records`

| Link | Stored where |
|---|---|
| Dataset version / sha256 / path / size / git commit | version tags `flightops.dataset_version`, `flightops.dataset_sha256`, `flightops.git_commit`; run artifact `config/dataset_fingerprint.json`; `dvc:<md5>` when a `.dvc` file exists |
| Validation report | parent run artifact `data/data_quality_report.json` |
| Preprocessing / FE | run artifact `config/preprocessing_config.json`; tag `flightops.feature_engineering_version` |
| Training config | run artifact `config/training_config.json` |
| Experiment / run | tags `flightops.experiment_id`, `flightops.run_id`, `flightops.parent_run_id` |
| Model artifact | `ModelVersion.source` |
| Approval / deployment | tags `approved_by/at`, `deployed_by/at`; `artifacts/governance/events.jsonl` (append-only) |
| Prediction records | `artifacts/monitoring/predictions.db`, joined on `model_version` |

The lineage page (`Model Lineage`) and `flightops_ml.tracking.lineage` read exactly these sources. A link that cannot be found is reported as **missing** with the reason; it is never synthesised.

## Versioning the dataset with DVC

```powershell
git init; dvc init
dvc add data/raw/ai4i2020.csv          # creates ai4i2020.csv.dvc (commit it; the CSV itself is git-ignored)
dvc remote add -d localstore ..\dvc-store
dvc push
dvc repro                               # runs validate -> train (see dvc.yaml)
```
