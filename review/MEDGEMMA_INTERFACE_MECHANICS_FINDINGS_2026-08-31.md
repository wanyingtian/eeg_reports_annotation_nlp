# MedGemma input-interface mechanics: completed bounded replay

Author-working engineering evidence, 2026-08-31. Not another protected-cohort
evaluation or a new performance row. The existing study predictions are unchanged.

## What was established

The native chat call can be reproduced through ordinary completion by explicitly
assembling its native input token sequence, while retaining Chris's classification
task and output grammar. There is no need to replace the structured pipeline
with unconstrained chat or a remote service.

The fixed protocol is [the diagnostic specification](MEDGEMMA_INTERFACE_MECHANICS_DIAGNOSTIC.md).
The [aggregate receipt](model-receipts/medgemma-interface-mechanics-2026-08-31.json)
binds the exact model, template, task, grammar, source artifacts, runtime versions
and call-receipt set. Producing code: `f86084915960f538de0d7e0e8fb2a6f8592c4c8f`.

Eight fixed, spaced positions from the existing first-100 Zoe development
manifest were each replayed in five arms. All 40 calls completed locally; all
40 actual input sequences were verified, and all outputs were schema-valid.
Every call ended normally, with 66--78 completion tokens. Summed generation
time was 1,194.84 seconds (about 20 minutes); this is not a comparative benchmark.

| Comparison on the same eight reports | Identical five-label predictions | Identical answer text |
| --- | ---: | ---: |
| Native chat / manually assembled completion | 8/8 | 8/8 |
| Assembled native / historical stopping restored | 8/8 | 8/8 |
| Historical raw / outer whitespace trimmed | 6/8 | 1/8 |
| Historical raw / native chat | 5/8 | 0/8 |
| Trimmed raw / native chat | 5/8 | 0/8 |

Both historical raw and native-chat replays reproduced their respective saved
five-label predictions on all eight reports. These are ordinal-label identity
counts, not correct-answer counts; reference labels were never loaded.

## Operational explanation, with limits

The installed chat handler renders its embedded template, tokenizes the result
with special tokens enabled and without adding a second BOS token, and calls
the same completion engine. The exact one-user-turn structure tested here is:

```text
<bos><start_of_turn>user
[classification task + report, outer whitespace trimmed]<end_of_turn>
<start_of_turn>model
```

The manual route's token IDs matched the captured native request before
generation and the actual evaluated token sequence afterward. This establishes
input/transport equivalence in this replay. Native stopping behavior was not
needed to reproduce these eight outputs. Whitespace does matter on some cases,
but trimming alone did not reproduce the native predictions on all cases.

Google documents the trained instruction-turn structure in its
[Gemma input-format guidance](https://ai.google.dev/gemma/docs/core/prompt-structure)
and the native template in the
[MedGemma model card](https://huggingface.co/google/medgemma-27b-text-it).
The measured behavior is consistent with using the model's expected input
format. This is not a causal account of individual hidden representations,
an ablation of every special token, a grammar-free comparison, or evidence of
full-cohort explanation quality. Cross-hardware bitwise determinism is not claimed.

## Existing saved same-report outputs

The diagnostic also joined the two already-saved development outputs on all
100 exact report keys. Forty reports differed in at least one ordinal label.
For focal epileptiform activity, 24 binary predictions changed from present in
the raw interface to absent in the native interface; none changed in the
reverse direction. That describes the prediction shift, not its correctness
or a new selected performance endpoint. All category counts remain in the receipt.

The local `same-report-comparison.html` contains the complete paired saved
outputs and source reports, with expandable replay prompts and receipts for
the eight selected cases. It remains in the ignored governed run directory;
do not attach it to the author email or copy its cases into a public PDF.

## Reproduction and boundaries

`scripts/diagnose_medgemma_interface.py` validates the frozen input/code
contract and checkpoints on resume. A completed run returns without loading
the model or regenerating calls. `scripts/export_medgemma_interface_diagnostic.py`
revalidates inputs and completed receipts, exports a positive allowlist of
aggregate fields, and refuses incomplete or changed frozen results. Use its
`--check` option to verify an existing export. The export retains invalid
outcomes if present rather than silently excluding them.

No protected evaluation was rerun; no configuration was selected; no submitted,
reproduced, v1, v2, v2.1 or external v5g result was overwritten. The eight cases
are development engineering checks, not an accuracy sample. Existing favorable
and adverse full-cohort findings remain intact.
