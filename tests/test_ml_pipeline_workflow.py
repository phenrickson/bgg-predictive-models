# tests/test_ml_pipeline_workflow.py
from pathlib import Path

import yaml

WF_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
STAGES = [
    "run-generate-text-embeddings.yml", "run-complexity-scoring.yml", "run-scoring-service.yml",
    "run-generate-embeddings.yml", "run-collection-scoring.yml", "build-collection-reports.yml",
]
OLD_EVENTS = ["text_embeddings_complete", "complexity_complete", "embeddings_complete"]


def _load(name):
    wf = yaml.safe_load((WF_DIR / name).read_text(encoding="utf-8"))
    return wf, (wf[True] if True in wf else wf["on"])


def test_orchestrator_trigger_and_order():
    wf, on = _load("ml-pipeline.yml")
    assert on["repository_dispatch"]["types"] == ["dataform_complete"]
    jobs = wf["jobs"]
    chain = ["text-embeddings", "complexity", "scoring", "game-embeddings", "collection-scoring"]
    for prev, job in zip(chain, chain[1:]):
        assert jobs[job]["needs"] == prev
    assert jobs["collection-reports"]["needs"] == "collection-scoring"
    # Collection scoring must finish (so its predictions publish the same day) but
    # its failure must not block publish: only the game-level chain gates it.
    notify = jobs["notify-warehouse"]
    assert notify["needs"] == ["game-embeddings", "collection-scoring"]
    assert "!cancelled()" in notify["if"]
    assert "needs.game-embeddings.result == 'success'" in notify["if"]
    assert "collection-scoring.result" not in notify["if"]
    assert "ml_complete" in str(jobs["notify-warehouse"])
    for job in chain + ["collection-reports"]:
        assert jobs[job]["uses"].startswith("./.github/workflows/")
        assert jobs[job]["secrets"] == "inherit"


def test_stages_are_callable_and_have_no_old_triggers():
    for name in STAGES:
        wf, on = _load(name)
        text = (WF_DIR / name).read_text(encoding="utf-8")
        assert "workflow_call" in on, name
        assert "workflow_dispatch" in on, name
        assert "repository_dispatch" not in on and "workflow_run" not in on and "schedule" not in on, name
        for event in OLD_EVENTS:
            assert event not in text, (name, event)


def test_service_images_rebuild_on_src_changes():
    # The images bundle src/, so ml_inputs/loader changes must redeploy them.
    for name in ["docker-scoring-build.yml", "docker-collections-build.yml", "docker-embeddings-build.yml"]:
        _, on = _load(name)
        assert "src/**" in on["push"]["paths"], name
