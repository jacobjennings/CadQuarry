# Published seed lists

Each `v*.toml` here is a versioned list of **canonical reproducible corpora**.
A corpus is fully determined by the generator version plus a seed, count, and
config — so this list (with the generator) *is* the deliverable; any published
dataset is just a convenience artifact you can regenerate at will.

## Reproduce a corpus

Pick an entry from `v1.toml` and run its seed/count/config:

```bash
# e.g. the "smoke" entry
cadquarry generate --seed 42 --count 100 \
    --config configs/default.toml --out datasets/smoke/

# confirm it re-executes and signatures hold
cadquarry verify datasets/smoke/
```

Same `generator_version` + same seed + same config ⇒ identical `manifest.jsonl`
and identical geometry signatures, bit-for-bit.

## Versioning

`[meta].generator_version` and `[meta].signature_version` pin the exact
generator and geometry-signature definition a list targets. Reproducing a
corpus requires matching `cadquarry.__version__` and
`filter.SIGNATURE_VERSION`; mismatches can change which parts count as
duplicates (and therefore the corpus). When the generator changes in a way
that affects output, publish a new `v{N}.toml` rather than editing an old one.
