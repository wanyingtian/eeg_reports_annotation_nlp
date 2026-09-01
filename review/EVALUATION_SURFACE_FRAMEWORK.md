# Evaluation-surface framework

## Why this exists

“Mistral” and “MedGemma” are model-family names, not complete experiments. A
reported value can also depend on the exact weight artifact and quantization,
model-facing interface, prompt, grammar, population, reference annotator,
selection history, and endpoint. If those factors move together, calling the
result a model comparison hides what actually changed.

The typed registry in
[`model-receipts/jbhi-evaluation-surface-registry.json`](model-receipts/jbhi-evaluation-surface-registry.json)
makes each factor explicit. It integrates the useful categories in the supplied
MedGemma summaries without copying provisional scores or treating an external
summary as a validated producing bundle.

Validate it with:

```bash
make verify-evaluation-surfaces
```

The validator checks design semantics only. It does not read report text,
predictions, patient identifiers, or model weights, and it cannot admit a result
to the manuscript.

## Three comparison types readers should not confuse

### 1. Controlled ablation: what did this one change do?

Hold the model artifact, task, prompt, grammar, population, reference, and
metrics fixed. Change exactly one of interface, prompt, grammar, or
quantization. The completed first-100 raw-versus-native interface checks for
MedGemma and Mistral belong here.

This is the strongest design for explaining interface sensitivity. It does not
establish performance on the protected evaluation population.

### 2. Model-native task comparison: how do the models perform when each is
used as intended?

Hold the task semantics, population, reference, and endpoints fixed. Give each
instruction-tuned model its receipted native turn serialization while retaining
the structured output contract. This is fairer operationally than forcing every
model through another model's serialization.

It is still a comparison of complete model artifacts. Differences cannot be
attributed only to pretraining or base weights because tokenizer, architecture,
quantization, and native serialization also differ.

### 3. Configured-system comparison: which frozen system works better here?

Compare complete frozen configurations on the same cases and reference. Prompt,
grammar, interface, quantization, and model may all differ. This is useful for a
deployment or “best available configuration” question, but it is not a causal
model-family experiment.

The current historical-Mistral versus native-MedGemma evaluation is correctly
registered this way. The supplied v5g comparison can enter the same class after
exact intake.

## The symmetric interface safeguard

Two asymmetric comparisons can both be misleading:

- forcing MedGemma through the historical raw-completion envelope can
  understate MedGemma; and
- comparing a prompt- and grammar-developed MedGemma configuration only with
  historical Mistral can understate what Mistral would do under an equally
  model-native treatment.

The registry therefore keeps the submitted study immutable, records the two
small development interface ablations, labels the current full comparison as a
configured-system contrast, and reserves a separately named native-Mistral
evaluation surface as **planned, not run**. Registering that possible surface
does not authorize or require another computation.

## Categories brought in from the external summaries

The registry can now represent:

- prompt versions v1 through v10 and the v5g configuration name;
- grammar-enforced consistency as a factor separate from prompt content;
- Q2 and Q4 quantizations;
- pooled, Zoe, Maria, and large unlabeled populations;
- LD reference and SG sensitivity surfaces;
- per-label binary (“Core”) F1;
- strict four-level endpoints;
- all-five-label whole-report exact match; and
- unlabeled positive-call rates and model-to-model agreement.

The external phrase “Certainty F1” is deliberately registered as
`exact_four_level_f1_external_formula_pending`. Exact-match accuracy and F1 are
not interchangeable. The producing bundle must identify the multiclass or
class-specific F1 construction before that endpoint can be reproduced or
combined with native analyses.

The external prompt-development grid is represented as a **design family**, not
as ten admitted result surfaces. That retains its valuable processing structure
while preserving the source-intake gate for exact prompts, grammar, runtime,
selection history, manifests, predictions, and formulas.

## Population rules

Every surface states whether it includes development reports and reconciles its
component counts. In particular:

- the native held-out evaluation is 1,395 Zoe plus 499 Maria reports = 1,894;
- the supplied prompt summaries describe 1,495 Zoe plus 499 Maria reports =
  1,994, which includes the first 100 Zoe development reports; and
- another supplied summary describes 2,493 reference-labeled reports and a
  45,545-report largely unlabeled corpus.

The difference between 1,994 and 2,493 is 499, but this is only a diagnostic
clue. The registry names the 2,493 surface as unreconciled and does not infer its
selection history or identity. A pooled 1,994 prompt-selection score is likewise
exploratory unless exact history demonstrates a separate selection/evaluation
boundary.

## Metric and claim rules

- Reference-based metrics cannot be assigned to an unlabeled surface.
- Positive-call prevalence and model agreement on unlabeled reports are
  descriptive; neither identifies which model is clinically correct.
- Whole-report exact match is an accuracy, not an F1 score.
- Per-label F1 should be accompanied by TP, FP, FN, TN and operating-point
  measures because it omits true negatives.
- A controlled ablation must declare exactly the one factor it changes.
- A configured-system or model-native task contrast must state that it does not
  isolate a base-weight effect.
- A held-out surface cannot silently include development cases.
- Public design records cannot contain observed metric values.

## Adding a new result without drifting the evaluation

1. Validate its producing bundle through the existing typed intake contract.
2. Add one surface with all ten factors resolved, exact population arithmetic,
   provenance node, artifact revision, and registered endpoints.
3. Add a contrast only when both surfaces use the same report keys and the
   intended factor differences are explicit.
4. Classify the contrast as an ablation, model-native task comparison,
   configured-system comparison, cohort stratification, reproduction, or
   unlabeled description.
5. Keep keyed predictions and report/patient identifiers in governed storage.
6. Store approved aggregate results in the result ledger, not this design
   registry.

This makes onboarding another local model an ordinary extension of Chris's
model-agnostic evaluation framework while preserving the submitted Mistral
study and every post-submission configuration as distinct evidence.
