import pytest

from flightops_ml.registry import audit, promotion
from flightops_ml.registry.registration import (
    ALIAS_CANDIDATE,
    STATUS_TAG,
    get_champion,
    versions_table,
)
from flightops_ml.training.train import run_training_pipeline
from tests.conftest import FAST_OVERRIDES

ONE_MODEL = {**FAST_OVERRIDES, "models": {**FAST_OVERRIDES["models"], "enabled": ["random_forest"]}}


def test_candidate_registered_with_status_alias_and_tags(trained_session):
    c, name, v = trained_session["client"], trained_session["name"], trained_session["version"]
    mv = c.get_model_version(name, v)
    assert mv.tags["flightops.threshold"] and mv.tags["flightops.dataset_sha256"] and mv.tags["flightops.run_id"]
    assert mv.tags[STATUS_TAG] in {"VALIDATED", "APPROVED", "DEPLOYED"}
    assert "gate_report" in " ".join(mv.tags)


def test_deployed_version_is_champion_and_candidate_alias_released(trained_session):
    c, name, v = trained_session["client"], trained_session["name"], trained_session["version"]
    assert str(get_champion(c, name).version) == v
    assert c.get_model_version(name, v).tags[STATUS_TAG] == "DEPLOYED"
    assert ALIAS_CANDIDATE not in c.get_registered_model(name).aliases


def test_cannot_skip_approval_or_deploy_unapproved(trained_session):
    c, name = trained_session["client"], trained_session["name"]
    res = run_training_pipeline(ONE_MODEL, register=True, trigger="pytest-2")
    v2 = res.registered_version
    with pytest.raises(promotion.PromotionError):
        promotion.deploy(c, name, v2, "someone")  # not approved
    if c.get_model_version(name, v2).tags[STATUS_TAG] == "VALIDATED":
        with pytest.raises(promotion.PromotionError):
            promotion.approve(c, name, v2, "", "no approver")
    # champion untouched by a newly trained model
    assert str(get_champion(c, name).version) == trained_session["version"]


def test_full_promotion_then_rollback(trained_session):
    c, name, v1 = trained_session["client"], trained_session["name"], trained_session["version"]
    res = run_training_pipeline(ONE_MODEL, register=True, trigger="pytest-3")
    v2 = res.registered_version
    status = c.get_model_version(name, v2).tags[STATUS_TAG]
    if status != "VALIDATED":
        pytest.skip(f"candidate {v2} did not pass the gates ({status}); promotion path covered elsewhere")
    promotion.approve(c, name, v2, "qa-lead", "better on holdout")
    promotion.deploy(c, name, v2, "ops")
    assert str(get_champion(c, name).version) == v2
    assert c.get_model_version(name, v1).tags[STATUS_TAG] == "ARCHIVED"
    restored = promotion.rollback(c, name, "ops", "regression found")
    assert restored == v1 and str(get_champion(c, name).version) == v1
    assert c.get_model_version(name, v2).tags[STATUS_TAG] == "ARCHIVED"
    events = [e["event"] for e in audit.read_events()]
    assert {"register", "validation", "approval", "deploy", "rollback"} <= set(events)
    with pytest.raises(promotion.PromotionError):
        promotion.rollback(c, name, "ops", "again")  # nothing further to roll back to


def test_cannot_archive_champion(trained_session):
    c, name = trained_session["client"], trained_session["name"]
    with pytest.raises(promotion.PromotionError):
        promotion.archive(c, name, str(get_champion(c, name).version), "oops")


def test_versions_table_for_dashboard(trained_session):
    df = versions_table(trained_session["client"], trained_session["name"])
    assert {"version", "business_status", "champion", "test_pr_auc", "threshold"} <= set(df.columns)
    assert df["champion"].sum() == 1
