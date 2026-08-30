# Model-native interface sensitivity protocol

## Purpose

The submitted study, the completed Mistral reproduction, and the completed
matched-historical MedGemma comparator used Chris's historical raw-completion
serialization. The later MedGemma development sensitivity changed only the
model-facing serialization to the GGUF's embedded chat template and improved
core accuracy on the fixed 100-report Zoe development set. It did **not**
remove the classification grammar.

This protocol asks two bounded post-submission questions:

1. Does the pinned Mistral-7B-Instruct-v0.2 artifact also respond differently
   when the unchanged classification task is presented through its native
   instruction template?
2. Can the thesis-derived explanation stage use the same model-native envelope
   while retaining its exact prompt, grammar, fixed classification JSON, and
   verbatim-evidence requirement?

Neither question changes the submitted result. All outcomes are retained.

## Stable methodological core

The grammar remains the model-independent output contract. A chat template is
an input adapter: it adds the role and turn-control tokens used during a
model's instruction tuning. These two controls operate on different sides of
the generation:

```text
fixed task bytes -> model-native turn envelope -> local model -> GBNF output contract
```

For MedGemma, the successful native development run still used
`result_grammar.gbnf` with SHA-256
`5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016`.
The raw and native MedGemma development runs used the same model artifact,
classification prompt, grammar, deterministic sampling settings, source
database, and 100 report keys. The observed difference therefore isolates
serialization within the limits of one development cohort; it does not show
that grammar constraints were harmful.

## Experiment A: Mistral classification interface

The single candidate is preregistered in
`model-receipts/mistral-native-interface-sensitivity.preregistered.json`.

### Fixed

- exact `Mistral-7B-Instruct-v0.2` Q5_K_M GGUF and artifact hash;
- first 100 Zoe development reports in their frozen order;
- historical classification prompt and task-message bytes;
- four-level classification grammar;
- context, temperature, top-k, top-p, stop, and maximum-token settings; and
- five task definitions and all downstream metric semantics.

### Changed

Only the embedded Mistral chat template is applied. The pinned GGUF identifies
the template as `mistral-instruct` and serializes one user turn with
`[INST] ... [/INST]`. Its exact template SHA-256 is
`26a59556925c987317ce5291811ba3b7f32ec4c647c400c6cc7e3a9993007ba7`.

### Development analysis

The native output is first frozen using the same result-blind structural rule
used for MedGemma: exact key coverage, valid four-level grammar outputs, no
runtime/context failure, and more than one full output pattern. Reference
metrics are computed only after that freeze. The paired comparator is the
exact-key subset of the completed fresh raw-completion Mistral reproduction,
not a newly optimized baseline.

Report all five core and exact-four-level results, paired changes with
intervals, exact McNemar tests with the existing multiplicity rule, output
distributions, latency, and every unfavorable result. A development benefit
does not authorize or imply a protected-evaluation run.

## Experiment B: explanation-interface isolation

The thesis correctly states that LLM self-explanations do not expose the
model's internal decision process and do not establish causality. This layer is
therefore named **self-prompted evidence extraction**. Its intended claims are
traceability to report text and alignment with a fixed model output.

For each model tested, hold one completed, frozen classification JSON artifact
constant. Feed the exact same report, classification JSON, explanation prompt,
and `result_grammar_exp.gbnf` through two explanation envelopes:

1. historical raw completion; and
2. the model's embedded native chat template.

The native route still passes the explanation grammar to llama.cpp. It does
not request or retain hidden chain-of-thought. The comparison reports:

- completion and JSON/grammar validity;
- exact decision-copy agreement with the fixed classification JSON;
- exact-substring traceability of non-fallback evidence phrases;
- the thesis's factuality and polarity-alignment measures, clearly labeled
  where fuzzy, semantic, or learned proxies are used;
- fallback frequency and evidence-count distribution; and
- paired latency and token counts.

No explanation metric is used to relabel a classification. No explanation is
described as proof of the model's causal decision process.

The resumable governed runner is
`scripts/run_fixed_classification_explanations.py`. It verifies checksums for
the source database, fixed prediction artifact, and ordered manifest before it
loads local model weights; refuses mixed-interface or out-of-order resumes;
retains invalid outputs; and emits only aggregate traceability counters in its
receipt. The keyed evidence CSV remains governed.

## Execution order and compute isolation

1. Let the active, frozen MedGemma protected-evaluation job finish without
   changing its worktree or competing for unified memory.
2. Run Mistral native classification once on the governed 100-report
   development manifest. Expected model time is about 15--25 minutes based on
   the completed raw Mistral reproduction.
3. Freeze structural validity before opening reference metrics.
4. Run the fixed-classification explanation comparison. Two Mistral
   explanation surfaces are expected to require roughly one hour in total.
5. Add a MedGemma explanation comparison only as a separately receipted
   development sensitivity; do not reinterpret its classification result.
6. Decide with the authors whether any frozen configuration merits protected
   evaluation and manuscript placement.

All inference remains local through `llama-cpp-python`; local-only model
resolution prevents a model-registry lookup during governed execution. Report
text, keys, reference labels, predictions, and evidence strings remain in
governed storage. Public-safe records contain configurations, hashes, counts,
and approved aggregate results only.

## Interpretation if Mistral also improves

The minimally invasive conclusion would be that instruction-tuned local models
should receive the unchanged EEG task through their own receipted instruction
serialization, while the grammar-constrained output schema, confidence labels,
evaluation cohorts, and thesis-derived evidence checks remain stable. The
submitted raw-completion Mistral study remains historical source evidence; the
native Mistral result is a named post-submission sensitivity.

If Mistral does not improve, retain that result. It would show that MedGemma's
interface sensitivity is model-specific rather than a universal pipeline
effect.
