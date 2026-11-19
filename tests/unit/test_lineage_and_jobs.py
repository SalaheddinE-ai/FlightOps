from flightops_ml.tracking.lineage import CHAIN, assemble_lineage
from flightops_ml.training import jobs

FULL_TAGS = {f"flightops.{k}": v for k, v in {
    "dataset_version": "sha256:abc", "dataset_sha256": "abc" * 20, "git_commit": "deadbeef",
    "feature_engineering_version": "1.0.0", "experiment_id": "1", "run_id": "r1", "parent_run_id": "p1",
    "threshold": "0.4", "business_status": "DEPLOYED", "deployed_at": "2026-01-01"}.items()}


def test_full_lineage_has_every_node_in_order():
    g = assemble_lineage(name="m", version="3", tags=FULL_TAGS, source="models:/m-1", run_params={"dataset_path": "x.csv"},
                         run_artifacts=["config/preprocessing_config.json", "config/training_config.json"],
                         parent_artifacts=["data/data_quality_report.json"],
                         deploy_events=[{"event": "deploy"}], n_predictions=12)
    assert [n["id"] for n in g["nodes"]] == CHAIN
    assert g["missing"] == []
    assert len(g["edges"]) == len(CHAIN) - 1


def test_missing_links_are_reported_not_fabricated():
    g = assemble_lineage(name="m", version="1", tags={"flightops.run_id": "r1"}, source=None, run_params=None,
                         run_artifacts=[], parent_artifacts=[], deploy_events=[], n_predictions=0)
    assert {"dataset", "validation_report", "deployment", "predictions", "model_artifact"} <= set(g["missing"])
    miss = next(n for n in g["nodes"] if n["id"] == "dataset")
    assert miss["status"] == "missing" and miss["reason"]


def test_job_state_roundtrip_and_progress(isolated_root):
    jid = jobs.create_job({"models": ["xgboost"]})
    assert jobs.read_job(jid)["status"] == "queued"
    p = jobs.JobProgress(jid, min_interval=0)
    p.stage("training xgboost")
    p.run("run123", model="xgboost", experiment="1")
    p.iteration(7, {"val_logloss": 0.2})
    st = jobs.read_job(jid)
    assert st["stage"] == "training xgboost" and st["iteration"] == 7 and st["run_id"] == "run123"
    assert jobs.list_jobs()[0]["job_id"] == jid
    assert jobs.read_job("missing") is None
