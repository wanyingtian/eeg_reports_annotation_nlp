from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from eeg_review.analysis_plan import build_comparison_readiness
from eeg_review.intake import EvidenceLayer, validate_intake
from eeg_review.manifest import sha256_file

LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def empty_artifact() -> dict[str, None]:
    return {"path": None, "sha256": None}


def make_provenance(
    layer: EvidenceLayer,
    model: Path,
    history: Path,
) -> dict[str, Any]:
    graph = f"jbhi-02463/comparator/{layer.value}"
    root = f"{graph}/configuration/test"
    weights_revision = "1" * 40
    producing_revision = "2" * 40
    evaluation_revision = "3" * 40
    integration_revision = "4" * 40
    integrated_revision = "5" * 40
    weights = f"{graph}/weights/test@{weights_revision}"
    producing = f"{root}/producing-bundle/test"
    evaluation = f"{root}/evaluation/test-manifest-hash"
    integration = f"{root}/integration/jbhi-revision-toolchain@test-revision"
    return {
        "graph_id": graph,
        "root_node_id": root,
        "admission": {
            "status": "integrated",
            "producing_bundle_received": True,
            "validated": True,
            "author_group_admitted": True,
            "integrated": True,
            "transition_receipt": artifact(history),
        },
        "nodes": [
            {
                "node_id": weights,
                "node_type": "upstream_weights_quantization",
                "status": "source_of_record",
                "repository_id": "example/model",
                "revision": weights_revision,
                "artifact": artifact(model),
                "parents": [],
                "distribution_status": "weights_not_redistributed",
            },
            {
                "node_id": producing,
                "node_type": "producing_configuration",
                "status": "author_group_admitted",
                "repository_id": "example/producing-bundle",
                "revision": producing_revision,
                "artifact": artifact(history),
                "parents": [weights],
                "distribution_status": "receipt_only",
            },
            {
                "node_id": evaluation,
                "node_type": "inherited_evaluation_framework",
                "status": "source_of_record",
                "repository_id": "example/evaluation-framework",
                "revision": evaluation_revision,
                "artifact": empty_artifact(),
                "parents": [],
                "distribution_status": "source_config_publishable",
            },
            {
                "node_id": integration,
                "node_type": "integration",
                "status": "integrated",
                "repository_id": "example/integration",
                "revision": integration_revision,
                "artifact": empty_artifact(),
                "parents": [evaluation],
                "distribution_status": "source_config_publishable",
            },
            {
                "node_id": root,
                "node_type": "integrated_configuration",
                "status": "integrated",
                "repository_id": "example/integration",
                "revision": integrated_revision,
                "artifact": empty_artifact(),
                "parents": [weights, producing, evaluation, integration],
                "distribution_status": "source_config_publishable",
            },
        ],
        "role_assertions": [
            {
                "assertion_id": "weights-originator",
                "node_id": weights,
                "role": "originator",
                "party": "upstream-test-party",
                "status": "confirmed",
                "scope": "test upstream artifact",
                "revision": "6" * 40,
                "evidence_artifact": empty_artifact(),
            },
            {
                "assertion_id": "configuration-originator",
                "node_id": producing,
                "role": "originator",
                "party": "configuration-test-party",
                "status": "confirmed",
                "scope": "test producing configuration",
                "revision": "7" * 40,
                "evidence_artifact": empty_artifact(),
            },
            {
                "assertion_id": "configuration-received-from",
                "node_id": producing,
                "role": "received_from",
                "party": "configuration-test-party",
                "status": "confirmed",
                "scope": "test producing-bundle transfer",
                "revision": "8" * 40,
                "evidence_artifact": empty_artifact(),
            },
            {
                "assertion_id": "evaluation-originator",
                "node_id": evaluation,
                "role": "originator",
                "party": "evaluation-test-party",
                "status": "confirmed",
                "scope": "test evaluation framework",
                "revision": "9" * 40,
                "evidence_artifact": empty_artifact(),
            },
            {
                "assertion_id": "integration-contributor",
                "node_id": integration,
                "role": "contributor",
                "party": "integration-test-party",
                "status": "confirmed",
                "scope": "test integration",
                "revision": "a" * 40,
                "evidence_artifact": empty_artifact(),
            },
            {
                "assertion_id": "scientific-governance",
                "node_id": root,
                "role": "scientific_governance",
                "party": "governance-test-party",
                "status": "confirmed",
                "scope": "test scientific admission",
                "revision": "b" * 40,
                "evidence_artifact": empty_artifact(),
            },
        ],
        "distribution_status": {
            "ancestry_receipt": "receipt_only",
            "source_configuration": "source_config_publishable",
            "outputs": "outputs_governed",
            "weights": "weights_not_redistributed",
        },
        "integration_transfers_ownership": False,
        "scientific_governance_note": (
            "Integration records lineage and does not transfer ownership."
        ),
    }


def make_contract(tmp_path: Path, layer: EvidenceLayer) -> Path:
    model = tmp_path / "model.gguf"
    prompt = tmp_path / "prompt.txt"
    history = tmp_path / "prompt-history.json"
    grammar = tmp_path / "grammar.gbnf"
    template = tmp_path / "chat-template.txt"
    for path, content in (
        (model, b"model"),
        (prompt, b"prompt"),
        (history, b"{}"),
        (grammar, b"root ::= 'ok'"),
        (template, b"{{ messages }}"),
    ):
        path.write_bytes(content)

    manifest = tmp_path / f"{layer.value}-cohort.csv"
    predictions = tmp_path / f"{layer.value}-predictions.csv"
    write_csv(
        manifest,
        ["report_key", "patient_key"],
        [["r1", "p1"], ["r2", "p1"], ["r3", "p2"]],
    )
    write_csv(
        predictions,
        ["report_key", *LABELS],
        [["r1", 1, 1, 1, 1, 1], ["r2", 3, 3, 3, 3, 3], ["r3", 4, 4, 4, 4, 4]],
    )
    payload = {
        "schema_version": 3,
        "status": "frozen",
        "evidence_layer": layer.value,
        "provenance": make_provenance(layer, model, history),
        "model_identity": {
            "upstream_repo_id": "example/model",
            "upstream_revision": "0123456789abcdef",
            "artifact": artifact(model),
            "size_bytes": model.stat().st_size,
            "quantization": "test",
            "license": "test-only",
            "license_terms_url": (
                "https://developers.google.com/health-ai-developer-foundations/terms"
                if layer == EvidenceLayer.POST_SUBMISSION_MEDGEMMA
                else "https://example.invalid/test-license"
            ),
            "license_notice": "test metadata; no legal conclusion",
        },
        "runtime": {
            "engine": "llama.cpp",
            "engine_version": "1.0",
            "engine_revision": "abcdef",
            "chat_template": {
                "mode": "file",
                "source": "explicit file",
                "artifact": artifact(template),
                "applied": True,
            },
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "temperature": 0,
            "top_k": 40,
            "top_p": 0.95,
            "seed": 7,
            "max_tokens": 100,
            "hardware": "test",
            "operating_system": "test",
        },
        "prompt": {
            "id": "locked-prompt",
            "artifact": artifact(prompt),
            "development_population": "historical development only",
            "reference_outcomes_inspected_during_selection": False,
            "frozen_before_final_evaluation": True,
            "stopping_rule": "one prespecified prompt",
            "selection_history_artifact": artifact(history),
        },
        "grammar": {
            "mode": "gbnf",
            "artifact": artifact(grammar),
            "purpose": "four-level output syntax",
        },
        "key_contract": {
            "report_key_column": "report_key",
            "report_key_namespace": "study-report-v1",
            "report_key_normalization": "exact_string",
            "patient_key_column": "patient_key",
            "patient_key_namespace": "study-patient-v1",
            "patient_key_semantics_confirmed": True,
        },
        "cohorts": [
            {
                "cohort_id": "evaluation",
                "role": "evaluation",
                "manifest": {"artifact": artifact(manifest), "table": "reports"},
                "population": {
                    "source_records": 4,
                    "candidate_records": 4,
                    "included_records": 3,
                    "excluded_records_by_reason": {"corrupt": 1},
                    "reference_complete_records": 3,
                    "prediction_expected_records": 3,
                },
                "predictions": {
                    "surface": {
                        "artifact": artifact(predictions),
                        "table": "classifications",
                    },
                    "report_key_column": "report_key",
                    "label_columns": {label: label for label in LABELS},
                    "invalid_records": 0,
                    "unfinished_records": 0,
                },
            }
        ],
        "privacy_boundary": "governed",
    }
    path = tmp_path / f"{layer.value}-intake.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_intake_validates_exact_keys_and_population(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)

    assert result["ready_for_analysis"] is True
    assert result["issues"] == []
    population = result["population_arithmetic"]["evaluation"]
    assert population["candidate_minus_included"] == 1
    keys = result["key_validation"]["evaluation"]
    assert keys["exact_same_case_surface"] is True
    assert keys["patient_grouping_ready"] is True
    assert len(keys["report_key_set_sha256"]) == 64
    assert len(keys["report_to_patient_mapping_sha256"]) == 64
    assert "r1" not in json.dumps(result)


def test_public_template_exposes_pending_ancestry_without_becoming_ready() -> None:
    template = (
        Path(__file__).parents[1] / "review/model-receipts/contemporary-llm-intake.template.json"
    )
    payload = json.loads(template.read_text(encoding="utf-8"))
    result = validate_intake(template, check_files=False)

    assert payload["schema_version"] == 3
    assert "owner" not in payload["provenance"]
    assert result["ready_for_analysis"] is False
    assert result["provenance_validation"]["admission_status"] == "external_pending_intake"
    assert result["provenance_validation"]["node_count"] == 8
    assert result["provenance_validation"]["role_assertion_count"] == 9


def test_intake_makes_missing_and_duplicate_keys_explicit(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    predictions = tmp_path / "post_submission_medgemma-predictions.csv"
    write_csv(
        predictions,
        ["report_key", *LABELS],
        [["r1", 1, 1, 1, 1, 1], ["r1", 3, 3, 3, 3, 3], ["r3", 4, 4, 4, 4, 4]],
    )
    payload["cohorts"][0]["predictions"]["surface"]["artifact"] = artifact(predictions)
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)
    keys = result["key_validation"]["evaluation"]
    assert result["ready_for_analysis"] is False
    assert keys["prediction_duplicate_report_keys"] == 2
    assert keys["missing_prediction_keys"] == 1
    assert keys["exact_same_case_surface"] is False
    assert all("r2" not in issue["message"] for issue in result["issues"])


def test_intake_rejects_unreconciled_population_arithmetic(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.REPRODUCED_MISTRAL)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["cohorts"][0]["population"]["included_records"] = 2
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=False)
    assert any(
        "included_records plus named exclusions" in issue["message"] for issue in result["issues"]
    )


def test_intake_requires_unfavorable_prediction_rows_to_be_counted(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    predictions = tmp_path / "post_submission_medgemma-predictions.csv"
    write_csv(
        predictions,
        ["report_key", *LABELS],
        [["r1", 1, 1, 1, 1, 1], ["r2", 3, 3, "", 3, 3], ["r3", 4, 4, 4, 4, 4]],
    )
    payload["cohorts"][0]["predictions"]["surface"]["artifact"] = artifact(predictions)
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)
    keys = result["key_validation"]["evaluation"]
    assert keys["incomplete_or_invalid_prediction_records"] == 1
    assert any(
        "invalid_records plus unfinished_records" in issue["message"] for issue in result["issues"]
    )


def test_three_layer_readiness_is_scaffolding_not_evaluation(tmp_path: Path) -> None:
    intake_paths = {layer: make_contract(tmp_path, layer) for layer in EvidenceLayer}
    result = build_comparison_readiness(
        intake_paths,
        tmp_path / "readiness",
        bundle_root=tmp_path,
    )

    assert result["analysis_started"] is False
    assert len(result["preregistered_comparisons"]) == 3
    assert all(item["ready"] for item in result["preregistered_comparisons"])
    assert all(
        layer["provenance_admission_status"] == "integrated"
        for layer in result["evidence_layers"].values()
    )
    for comparison in result["preregistered_comparisons"]:
        plan = comparison["cohorts"][0]
        assert plan["same_case_ready"] is True
        assert plan["patient_grouped_ready"] is True
    rendered = json.dumps(result)
    assert "accuracy" not in rendered
    assert "r1" not in rendered


def test_readiness_blocks_swapped_layer_identity(tmp_path: Path) -> None:
    intake_paths = {layer: make_contract(tmp_path, layer) for layer in EvidenceLayer}
    intake_paths[EvidenceLayer.SUBMITTED_MISTRAL] = intake_paths[
        EvidenceLayer.POST_SUBMISSION_MEDGEMMA
    ]

    result = build_comparison_readiness(
        intake_paths,
        tmp_path / "readiness",
        bundle_root=tmp_path,
    )

    submitted = result["evidence_layers"]["submitted_mistral"]
    assert submitted["contract_evidence_layer_matches"] is False
    assert submitted["ready_for_analysis"] is False
    assert result["preregistered_comparisons"][0]["ready"] is False


def test_patient_mapping_mismatch_blocks_grouped_not_same_case_plan(tmp_path: Path) -> None:
    intake_paths = {layer: make_contract(tmp_path, layer) for layer in EvidenceLayer}
    reproduced = intake_paths[EvidenceLayer.REPRODUCED_MISTRAL]
    payload = json.loads(reproduced.read_text(encoding="utf-8"))
    manifest = tmp_path / "reproduced_mistral-cohort.csv"
    write_csv(
        manifest,
        ["report_key", "patient_key"],
        [["r1", "p1"], ["r2", "p2"], ["r3", "p2"]],
    )
    payload["cohorts"][0]["manifest"]["artifact"] = artifact(manifest)
    reproduced.write_text(json.dumps(payload), encoding="utf-8")

    result = build_comparison_readiness(
        intake_paths,
        tmp_path / "readiness",
        bundle_root=tmp_path,
    )
    comparison = next(
        item
        for item in result["preregistered_comparisons"]
        if item["comparison_id"] == "submitted_vs_reproduced_mistral"
    )
    cohort = comparison["cohorts"][0]
    assert cohort["same_case_ready"] is True
    assert cohort["patient_grouped_ready"] is False
    mapping_gate = next(
        gate
        for gate in cohort["patient_grouped_gates"]
        if gate["gate"] == "report_to_patient_mapping_matches"
    )
    assert mapping_gate["passed"] is False


def test_frozen_provenance_rejects_omitted_parent_revision(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    root_id = payload["provenance"]["root_node_id"]
    root = next(node for node in payload["provenance"]["nodes"] if node["node_id"] == root_id)
    integration_id = next(parent for parent in root["parents"] if "/integration/" in parent)
    integration = next(
        node for node in payload["provenance"]["nodes"] if node["node_id"] == integration_id
    )
    integration["revision"] = None
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)

    assert result["ready_for_analysis"] is False
    assert any(
        issue["field"] == f"provenance.parent_revision[{integration_id}]"
        for issue in result["issues"]
    )


def test_frozen_provenance_rejects_omitted_parent_role_assertions(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    evaluation_id = next(
        node["node_id"]
        for node in payload["provenance"]["nodes"]
        if node["node_type"] == "inherited_evaluation_framework"
    )
    payload["provenance"]["role_assertions"] = [
        assertion
        for assertion in payload["provenance"]["role_assertions"]
        if assertion["node_id"] != evaluation_id
    ]
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)

    assert result["ready_for_analysis"] is False
    assert any(
        issue["field"] == f"provenance.role_assertions[{evaluation_id}]"
        for issue in result["issues"]
    )


def test_provenance_rejects_ambiguous_owner_and_skipped_admission(tmp_path: Path) -> None:
    contract = make_contract(tmp_path, EvidenceLayer.POST_SUBMISSION_MEDGEMMA)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["provenance"]["owner"] = "ambiguous"
    payload["provenance"]["admission"]["validated"] = False
    contract.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_intake(contract, bundle_root=tmp_path, check_files=True)

    assert result["ready_for_analysis"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert "provenance.owner" in fields
    assert "provenance.admission" in fields
