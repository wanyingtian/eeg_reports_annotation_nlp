# Quantitative claim ledger

No row is cleared by prose alone. A cleared claim needs authoritative inputs,
an executable receipt, a named effect measure, and an identified manuscript
destination. Null and unfavorable findings use the same gate.

| Claim / topic | Required receipt | Current gate |
|---|---|---|
| “Up to 14%” prompt gain | Frozen prompt versions, sample sizes, unrounded CAA, development/test separation, absolute and relative change | Blocked on historical prompts and outputs |
| “25% more likely” explanation error | Aligned/misaligned raw counts, report-vs-category unit, risk difference and ratio (or odds ratio), 95% CI and justified test | Blocked on source alignment table |
| Maria BoW abnormality core-to-certainty accuracy decrease | Recovered submitted matrix counts and named effect measure | Aggregate resolved: core `282/499 = 0.565130`, exact four-level `156/499 = 0.312625`; absolute decrease `0.252505` (25.25 percentage points), relative decrease `44.68%`; not “more than half” and not probabilistic calibration |
| “Near-human” | Annotator roles, independence/adjudication, uncertainty, bounded agreement wording | Clinical/team decision; single-annotator agreement is not ground truth |
| “Traditional NLP failed” | Complete BoW+LR/BERT+LR configurations, OOF and external results, rare-class support/CIs | Narrow to the tested low-resource configurations |
| Calibration | OOF/external probabilities, Brier/log loss/ECE and bins | Applicable to probability-bearing baselines, not automatically to generated four-level LLM labels |
| Patient independence | Stable patient-key semantics, cross-split counts, grouped execution | Blocked until custodian confirms the key |
| Ethics/secondary use | Approved H18-02728 consent/waiver, de-identification and secondary-use language | PI/REB-only evidence |

Each eventual machine-readable ledger row should additionally bind the input
checksums, Git revision, command, seed, output path, analysis population,
numerator/denominator, unrounded value, display value, and reviewer ID.
