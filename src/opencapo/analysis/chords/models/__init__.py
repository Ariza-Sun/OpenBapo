"""Deep-learning chord experts, each an adapter around a published model.

Every adapter maps the model's own label space into the shared vocabulary
(`chords.vocab`) and returns frame-wise `ChordPosteriors`. Model packages are
imported lazily so the rest of the pipeline runs without them.
"""
