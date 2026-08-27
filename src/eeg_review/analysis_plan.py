from __future__ import annotations

from pathlib import Path
from typing import Any

from .intake import ComparatorIntake, EvidenceLayer, load_intake, validate_intake
from .io import atomic_write_json
from .manifest import build_manifest

PREREGISTERED_PAIRS = (
    {
        "comparison_id": "submitted_vs_reproduced_mistral",
        "layer_a": EvidenceLayer.SUBMITTED_MISTRAL,
        "layer_b": EvidenceLayer.REPRODUCED_MISTRAL,
        "purpose": "reproduction agreement and sensitivity; not model superiority",
    },
    {
        "comparison_id": "submitted_mistral_vs_post_submission_medgemma",
        "layer_a": EvidenceLayer.SUBMITTED_MISTRAL,
        "layer_b": EvidenceLayer.POST_SUBMISSION_MEDGEMMA,
        "purpose": "post-submission contemporary comparator on the frozen submitted surface",
    },
    {
        "comparison_id": "reproduced_mistral_vs_post_submission_medgemma",
        "layer_a": EvidenceLayer.REPRODUCED_MISTRAL,
        "layer_b": EvidenceLayer.POST_SUBMISSION_MEDGEMMA,
        "purpose": "runtime-context sensitivity; cannot replace the submitted source of record",
    },
)


def _cohort_map(intake: ComparatorIntake) -> dict[str, Any]:
    return {cohort.cohort_id: cohort for cohort in intake.cohorts if cohort.cohort_id}


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"gate": name, "passed": passed, "detail": detail}


def build_comparison_readiness(
    intake_paths: dict[EvidenceLayer, Path],
    output_dir: Path,
    *,
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    validations = {
        layer: validate_intake(path, bundle_root=bundle_root, check_files=True)
        for layer, path in intake_paths.items()
    }
    intakes = {layer: load_intake(path) for layer, path in intake_paths.items()}
    layers: dict[str, Any] = {}
    for expected in EvidenceLayer:
        validation = validations.get(expected)
        intake = intakes.get(expected)
        identity_matches = bool(intake and intake.evidence_layer == expected)
        layers[expected.value] = {
            "provided": validation is not None,
            "contract_sha256": validation.get("contract_sha256") if validation else None,
            "contract_evidence_layer_matches": identity_matches,
            "ready_for_analysis": bool(
                validation and validation.get("ready_for_analysis", False) and identity_matches
            ),
            "blocker_count": (
                sum(issue["severity"] == "blocker" for issue in validation["issues"])
                if validation
                else None
            ),
        }

    comparisons: list[dict[str, Any]] = []
    for definition in PREREGISTERED_PAIRS:
        layer_a = definition["layer_a"]
        layer_b = definition["layer_b"]
        intake_a = intakes.get(layer_a)
        intake_b = intakes.get(layer_b)
        validation_a = validations.get(layer_a)
        validation_b = validations.get(layer_b)
        cohort_ids = sorted(
            set(_cohort_map(intake_a)) | set(_cohort_map(intake_b))
            if intake_a and intake_b
            else set()
        )
        cohort_plans: list[dict[str, Any]] = []
        for cohort_id in cohort_ids:
            cohort_a = _cohort_map(intake_a).get(cohort_id) if intake_a else None
            cohort_b = _cohort_map(intake_b).get(cohort_id) if intake_b else None
            receipt_a = validation_a["key_validation"].get(cohort_id) if validation_a else None
            receipt_b = validation_b["key_validation"].get(cohort_id) if validation_b else None
            gates = [
                _gate("cohort_present_in_both_layers", bool(cohort_a and cohort_b), cohort_id),
                _gate(
                    "both_intakes_validated",
                    bool(
                        validation_a
                        and validation_b
                        and layers[layer_a.value]["ready_for_analysis"]
                        and layers[layer_b.value]["ready_for_analysis"]
                    ),
                    "both producing bundles must pass typed intake validation",
                ),
                _gate(
                    "report_key_namespace_matches",
                    bool(
                        intake_a
                        and intake_b
                        and intake_a.key_contract.report_key_namespace
                        == intake_b.key_contract.report_key_namespace
                    ),
                    "same namespace is required before treating report keys as comparable",
                ),
                _gate(
                    "exact_report_key_set_matches",
                    bool(
                        receipt_a
                        and receipt_b
                        and receipt_a["exact_same_case_surface"]
                        and receipt_b["exact_same_case_surface"]
                        and receipt_a["report_key_set_sha256"] == receipt_b["report_key_set_sha256"]
                    ),
                    "computed inside the authorized bundle; no report keys are emitted",
                ),
                _gate(
                    "canonical_label_surface_matches",
                    bool(
                        cohort_a
                        and cohort_b
                        and set(cohort_a.predictions.label_columns)
                        == set(cohort_b.predictions.label_columns)
                    ),
                    "canonical label names, not source column names, are compared",
                ),
            ]
            patient_gates = [
                _gate(
                    "patient_key_semantics_confirmed",
                    bool(
                        intake_a
                        and intake_b
                        and intake_a.key_contract.patient_key_semantics_confirmed
                        and intake_b.key_contract.patient_key_semantics_confirmed
                    ),
                    "custodian-confirmed patient linkage is required",
                ),
                _gate(
                    "patient_key_namespace_matches",
                    bool(
                        intake_a
                        and intake_b
                        and intake_a.key_contract.patient_key_namespace
                        and intake_a.key_contract.patient_key_namespace
                        == intake_b.key_contract.patient_key_namespace
                    ),
                    "patient keys must share a stable namespace across layers",
                ),
                _gate(
                    "patient_keys_complete",
                    bool(
                        receipt_a
                        and receipt_b
                        and receipt_a["patient_grouping_ready"]
                        and receipt_b["patient_grouping_ready"]
                    ),
                    "patient keys must be non-missing on the paired report surface",
                ),
                _gate(
                    "report_to_patient_mapping_matches",
                    bool(
                        receipt_a
                        and receipt_b
                        and receipt_a["report_to_patient_mapping_sha256"]
                        and receipt_a["report_to_patient_mapping_sha256"]
                        == receipt_b["report_to_patient_mapping_sha256"]
                    ),
                    "the report-to-patient mapping must be identical across layers",
                ),
            ]
            cohort_plans.append(
                {
                    "cohort_id": cohort_id,
                    "same_case_gates": gates,
                    "same_case_ready": all(gate["passed"] for gate in gates),
                    "patient_grouped_gates": patient_gates,
                    "patient_grouped_ready": all(
                        gate["passed"] for gate in [*gates, *patient_gates]
                    ),
                    "planned_statistics": {
                        "same_case": [
                            "paired metric differences",
                            "paired bootstrap confidence intervals",
                            "report-level exact McNemar sensitivity analysis",
                        ],
                        "patient_grouped": [
                            "patient-cluster paired bootstrap confidence intervals"
                        ],
                    },
                }
            )
        comparisons.append(
            {
                "comparison_id": definition["comparison_id"],
                "layer_a": layer_a.value,
                "layer_b": layer_b.value,
                "purpose": definition["purpose"],
                "cohorts": cohort_plans,
                "ready": bool(cohort_plans)
                and all(plan["same_case_ready"] for plan in cohort_plans),
            }
        )

    result = {
        "schema_version": 1,
        "status": "readiness_only_no_evaluation_performed",
        "evidence_layers": layers,
        "preregistered_comparisons": comparisons,
        "analysis_started": False,
        "claim_boundary": (
            "This receipt checks producing-bundle and alignment gates only. It contains no model "
            "performance estimate and cannot identify an unreceipted producing configuration."
        ),
        "privacy_boundary": (
            "Aggregate validation counts and key-set digests only; governed keys and predictions "
            "remain in authorized storage."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "comparison_readiness.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "comparison-readiness",
            list(intake_paths.values()),
            {
                "layers": sorted(layer.value for layer in intake_paths),
                "bundle_root_supplied": bundle_root is not None,
            },
        ),
    )
    return result
