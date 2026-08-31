# MedGemma interface mechanics: bounded development diagnostic

2026-08-31. User-authorized technical investigation, not another evaluation.

The question is whether the native chat result can be reproduced by explicitly
assembling its model input and using the completion API with the same output
grammar. The lower-level native request is intercepted before generation, so
its actual token IDs and effective stopping/sampling parameters can be compared.
The manually assembled prompt must match its tokens exactly before replay.
Special tokens are tokenized explicitly, with one beginning-of-sequence token;
blindly inserting an already-prefixed string into a completion API can duplicate it.

The policy in `src/eeg_review/interface_diagnostic.py` fixes eight spaced
positions in the existing 100-report development manifest and five arms, for
at most 40 new classification calls:

1. Historical unwrapped task/report input.
2. Same unwrapped input with only outer whitespace trimmed.
3. Existing native chat interface.
4. Explicitly assembled, token-identical input through the completion API,
   with native effective sampling/stopping parameters.
5. Same assembled input with historical stopping behavior restored.

All arms retain the same Q2_K artifact, classification task, grammar, 4096
context, 256-token completion limit, greedy decoding and Metal profile. Model
state is reset before each call. No reference labels enter the diagnostic.
All outputs, including invalid or nonmatching ones, are retained. There is no
best-arm selection, automatic expansion, protected-cohort access or replacement
of completed study results. The eight cases are not an accuracy sample.

Saved predictions on all 100 development reports are joined by exact report
keys for a paired disagreement description. Full report text, prompts, keys,
token IDs and per-case predictions stay in ignored governed storage. A local
HTML comparison supports author inspection without distributing cases in PDFs.
Only aggregate diagnostics and report-free template illustrations may enter
author-working documentation, clearly identified as post-hoc engineering evidence.

The mechanism being tested is input serialization and transport equivalence,
not the model's internal neural reasoning. Google documents the trained
instruction-turn format and MedGemma's use of a chat template:

- [Gemma prompt structure](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [Official MedGemma model card](https://huggingface.co/google/medgemma-27b-text-it)

Each call is checkpointed with content hashes. Resume validates the contract,
source artifacts and each completed checkpoint. A `PAUSE` file stops between
calls; removing it and rerunning resumes. An `ECLIPSED.json` governance marker
in the output or parent run prevents execution and analysis. No weights or
governed products are pushed to Git. Inference resolves the pinned model from
local cache only, with Python outbound connection calls denied.
