# Preliminary grammar-draft evidence, not the canonical release comparison

The first independent6000step pair was run before inspection caught incorrect agreement in18new unknown-name teaching
questions and one development prompt. Files transfer-preliminary-legacy-v22.json and transfer-preliminary-balanced-v22.json
retain those raw outputs unchanged. Do NOT report them as the results of the final corrected corpus.

transfer-preliminary-grammar-correction-v22.patch is the exact data-only change from that draft to the corrected release.
To reproduce those historical trials, reverse this patch ONLY in a throwaway source copy before running the Python
transfer_learning_experiment helper, which reads raw JSONL. The runtime manifest will intentionally no longer match;
do NOT build/install that temporary draft or regenerate data afterward. Never apply this to an owner workspace.

Both full A/B trials were rerun on the corrected final corpus. Canonical results are transfer-legacy-v22.json and
transfer-balanced-v22.json, documented in LEARNING_RESULTS_V22_RU.md. No extra parameter/data tuning selected a winner;
the grammar repair was the only training-data difference between these two experiment rounds.
