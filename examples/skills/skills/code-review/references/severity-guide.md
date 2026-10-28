# Severity guide

- **critical**: wrong results, data loss, security holes, crashes on normal input.
- **warning**: fails on edge cases (empty input, None, zero), poor error handling, hidden performance traps.
- **suggestion**: naming, structure, idioms, missing tests. Safe to ignore without breaking anything.

When unsure between two levels, pick the lower one and say why in one sentence.
