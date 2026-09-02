# Configuration-aware, evidence-linked local language-model evaluation

## The contribution in one sentence

Chris's pipeline can be expressed as a durable clinical-text research method:
run a local model through an explicit task adapter, keep confidence visible,
separate evidence extraction from classification, verify what evidence comes
from the source, and preserve the complete path from report cohort to reviewed
claim.

This is a synthesis of the established thesis lanes and their implemented
revision extensions. It is not a claim of a new clinical device, a new
foundation model, or hidden-chain-of-thought access.

## The six linked lanes

| Lane | Plain-language question | Machine contract | Boundary |
|---|---|---|---|
| Report transport | Which report is being processed, and in which study split? | Unique report key, exact text hash, immutable cohort manifest, development/evaluation role | A report key is not a patient key |
| Configured classification | What five findings does this particular configured system assign? | Pinned model artifact, native or historical serializer, prompt, grammar, sampling, five four-level fields | Compare configured systems; do not silently attribute every difference to weights |
| Confidence | Is the model declaring an absent or present finding, and with which confidence tier? | Levels 1–2 map to absent, 3–4 to present; levels 1/4 are confident and 2/3 low-confidence | The four-level declaration is not a calibrated probability |
| Evidence | What source passages does a separate, stateless evidence call return? | Present, absent, and qualifying passage roles or the historical reason schema | Evidence text is not hidden reasoning or proof that it caused the decision |
| Traceability and review | Is a passage an unchanged quotation, a review candidate, or unresolved? | Exact substring verification first; normalized, fuzzy, and semantic stages only route review | Location is not entailment, clinical relevance, or factuality |
| Study and governance | May the result support a paper claim? | Frozen split, result-blind selection, paired complete-case analysis, all outcomes retained, qualified review, hashed receipts | Author admission and clinical interpretation remain separate decisions |

## Why MedGemma strengthens rather than replaces the framework

MedGemma demonstrates that the model-facing serializer is part of the
configured system. Applying its embedded chat template while retaining the
same report, five tasks, four-level semantics, grammar, deterministic sampling,
and evaluation harness materially improved the development transport result.
The corresponding v1 configuration was frozen before development execution and
then evaluated once on the disjoint 1,894-report surface.

That is exactly the thesis framework working as intended: the model underneath
can change, but the observable task, output lawfulness, evidence separation,
and evaluation discipline remain explicit. A deliberately mismatched adapter
is retained as an interface-ablation result, not called a weak MedGemma model.

## What is already showable

1. **Reproduction:** the submitted Mistral workflow runs locally outside BDH
   with its historical outputs kept distinct from the fresh reproduction.
2. **Model onboarding:** MedGemma v1 uses a pinned local artifact and a
   model-appropriate serializer without changing the five clinical-text tasks
   or four-level output contract.
3. **Held-out discipline:** development and evaluation are disjoint by report
   key and normalized text; no evaluation result selected the v1 configuration.
4. **Confidence as discourse:** each four-level output remains visible both as
   a core present/absent call and as the model's declared confidence tier.
5. **Evidence visibility:** a typed adapter spans the historical explanation
   artifact and saved Mistral/MedGemma development evidence while preserving
   the different evidence roles.
6. **Reviewability:** a governed source-first packet exposes the report before
   the saved decision and evidence, with every substantive reviewer field left
   blank until a qualified person assesses it.

The review packet currently exercises the full lens on the historical Mistral
surface. Saved first-20 MedGemma evidence demonstrates adapter transport only;
the completed full-cohort MedGemma classification run did not generate
explanations. That absence must remain visible rather than filled by inference.

## A compact publication formulation

The paper need not become a broad platform paper. A concise methods statement
can say that the study evaluates **configuration-aware local language-model
systems** under a common EEG annotation contract. The revision can then show:

- the submitted Mistral study as the historical contribution;
- the independent reproduction as evidence of continuity;
- MedGemma v1 as a held-out contemporary configured-system comparison; and
- confidence, source evidence, and traceability as separable visibility lenses
  rather than a single opaque explanation score.

Detailed receipts, interface ablations, unresolved quotations, and exploratory
v2/v2.1 development remain in the supplement or response. Clinical deployment,
patient-grouped inference, calibrated probabilities, and v5g performance stay
outside any claim until their own evidence gates are met.

## Reusable operational rule

Every future model joins by adding a configuration node—not by rewriting the
study. It must declare model artifact, serializer, prompt, grammar, sampling,
cohort, selection role, outputs, and ancestry. Development may alter a named
configuration; protected evaluation may measure it once but may not feed back
into that configuration. Evidence and review can be extended independently,
with all missing layers stated explicitly.
