# Reference-aligned evidence traceability: pre-execution protocol

**Status:** endpoints frozen before aggregate calculation. This is a
development-surface descriptive analysis, not a preregistered primary study,
clinical explanation validation or part of the current JBHI revision.

## Question

When each saved configured system agrees or disagrees with the retained
Reference Annotator label, how often does it also return substantive evidence
and how often is at least one evidence phrase an unchanged quotation from the
source report?

This analysis connects three already-existing layers without treating them as
interchangeable:

1. the Reference Annotator's four-level category label;
2. the configured system's saved four-level decision; and
3. the configured system's saved evidence phrases and source-location status.

## Frozen analysis surface

- Population: the exact first 100 Zoe Reference Annotator development reports,
  in the frozen manifest order.
- Units: 100 reports × five EEG categories = 500 units per configured system.
- Systems:
  - saved historical-interface Mistral classification and evidence outputs;
  - saved native-interface MedGemma v1 fixed classifications and evidence.
- Reference: the five retained four-level labels in the governed development
  database.
- No new model inference and no use of the 1,894 held-out reports.
- The separate 20-case blinded reader instrument and its unblinding key are not
  opened or used.

## Frozen endpoints

For every system, category and agreement stratum, retain counts and transparent
denominators for:

1. binary agreement with the Reference Annotator (levels 1–2 versus 3–4);
2. exact four-level agreement;
3. units with substantive evidence;
4. units with at least one unchanged source quotation;
5. evidence-bearing units whose every substantive segment is unchanged;
6. substantive segment counts partitioned into unchanged quotation,
   normalization/location candidate and unresolved; and
7. declared-no-evidence or otherwise empty units.

The aggregate must provide both all-unit and evidence-bearing denominators.
Results are reported overall, by category, by binary-agreement stratum and by
exact-agreement stratum. All configured systems and all strata are retained,
including empty or unfavorable cells.

## Interpretation contract

The permissible term is **reference-aligned evidence traceability**. It means
that a saved system decision can be compared with the retained reference label
and that its evidence can be checked for source location.

It does not establish:

- clinical correctness or diagnostic ground truth;
- entailment, sufficiency or clinical usefulness of a quoted phrase;
- causal faithfulness or hidden model reasoning;
- calibrated confidence;
- model-family or architecture superiority; or
- held-out or population-level performance.

Because the two configured systems use different evidence schemas, differences
cannot be attributed to model weights alone. No hypothesis test, confidence
interval or rank claim is planned on this development surface.

## Decision gate

After the complete result is retained, record one of three next steps:

1. stop at the development diagnostic;
2. seek the optional EEG-qualified 20-case review before scaling; or
3. run a separately frozen full-evaluation or model-by-schema experiment.

The choice must be based on whether the analysis answers a defined methods
question, not whether one configured system has the more favorable aggregate.
