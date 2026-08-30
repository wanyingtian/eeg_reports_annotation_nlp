# Quantitative claim ledger

No row is cleared by prose alone. A cleared claim needs authoritative inputs,
an executable receipt, a named effect measure, and an identified manuscript
destination. Null and unfavorable findings use the same gate.

| Claim / topic | Required receipt | Current gate |
|---|---|---|
| “Up to 14%” prompt gain | Frozen prompt versions, sample sizes, unrounded CAA, development/test separation, absolute and relative change | Chris recalled first-100 prompt development and later evaluation described as reports 100-1000, but not the percentage source; blocked on exact prompt/output artifacts and selection semantics |
| “25% more likely” explanation error | Aligned/misaligned raw counts, report-vs-category unit, risk difference and ratio (or odds ratio), 95% CI and justified test | Blocked on source alignment table |
| Maria BoW abnormality core-to-certainty accuracy decrease | Recovered submitted matrix counts and named effect measure | Aggregate resolved: core `282/499 = 0.565130`, exact four-level `156/499 = 0.312625`; absolute decrease `0.252505` (25.25 percentage points), relative decrease `44.68%`; not “more than half” and not probabilistic calibration |
| “Near-human” | Annotator roles, independence/adjudication, uncertainty, bounded agreement wording | Clinical/team decision; single-annotator agreement is not ground truth |
| Contemporary LLM improvement | Exact model/prompt/grammar receipts, immutable same-case predictions, declared prompt-selection population, complete matrices, paired intervals/tests, and patient grouping where possible | The independent matched-historical Q2 result is completed and preserved, including its unfavorable findings. One native-chat configuration was frozen by a result-blind rule on the first 100 Zoe reports; protected 1,395/499 evaluation is fully specified but awaits the documentary execution gate. Its finalizer will emit 90 source-hashed author-review claim rows without automatic manuscript admission. External v5g remains a distinct configuration pending exact intake. |
| Contemporary performance on the 45k corpus | Frozen report manifest, model receipts, descriptive agreement/prevalence tables, and governed clinical validation sample | Unlabeled disagreement is descriptive only; increased positive calls cannot be called correct without reference labels or qualified review |
| “Traditional NLP failed” | Complete BoW+LR/BERT+LR configurations, OOF and external results, rare-class support/CIs | Fresh OOF/external evidence complete: high rare-label accuracy sometimes has zero or near-zero F1. Narrow to the tested low-resource configurations; do not generalize to traditional NLP as a class |
| Calibration | OOF/external probabilities, Brier/log loss/ECE and bins | Completed for fresh Zoe/Maria BoW/BERT and exact submitted Maria rows; patient-cluster intervals and exact submitted Zoe prediction rows remain pending. Generated four-level LLM labels are ordinal outputs, not automatically calibrated probabilities. A new opt-in binary-core Mistral probability instrument and development-only four-level certainty mapper are implemented with explicit stability diagnostics, but no development threshold fit or evaluation result has been run |
| Patient independence | Stable patient-key semantics, cross-split counts, grouped execution | Blocked until custodian confirms the key |
| Ethics/secondary use | Approved H18-02728 consent/waiver, de-identification and secondary-use language | PI/REB-only evidence |

The machine-readable result ledger is generated with
`scripts/study_job.py ledger`. It binds each available unrounded estimate and
confidence interval to the aggregate source receipt's SHA-256, analysis ID,
population size, interval unit, bootstrap count, and available
numerator/denominator. Its own run manifest records the Git revision and input
hashes. Manuscript destination, display rounding, reviewer-comment ID, and
claim-clearance status remain deliberate authoring decisions and are not
inferred by the generator.

For the separately frozen MedGemma native-interface sensitivity,
`render_medgemma_native_author_bundle.py` converts a completed public-safe candidate into
a deterministic 20-row primary LaTeX table, 90-row full effect ledger, and bounded
methods/results/reviewer-response fragments. Without a hash-bound manuscript-admission
receipt, every fragment remains author-working. The admission validator refuses partial
primary-table selection, so null or unfavorable primary rows cannot be silently dropped.
