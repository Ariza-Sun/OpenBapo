"""Chord and key recognition: a committee of experts fused on the beat grid.

See docs/plans/step-2.md. `vocab` defines the shared label space, `fusion`
combines expert outputs, and the rule experts live in `chroma`, `bassline`
and `key`. Deep-learning experts are adapters under `models/`.
"""
