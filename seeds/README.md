# Published seed lists

Each `v*.toml` here is a versioned list of **canonical reproducible corpora**.
A corpus is fully determined by the generator version plus a seed, count, and
config — so this list (with the generator) *is* the deliverable; any published
dataset is just a convenience artifact you can regenerate at will.

- [`v1.toml`](v1.toml) — generator `0.4.0`; the original nine primitive
  families. Still valid for reproducing the originally published corpora.
- [`v2.toml`](v2.toml) — generator `0.5.0`; adds the build123d-backed `gear`
  and `threaded` families and bumps the CadQuery floor (OCP alignment). The
  pinned `mech` library versions are recorded under `[meta.mech]`. Because the
  new families (and the kernel bump) change the RNG-driven corpus for a given
  seed, this is a fresh generation rather than an edit of `v1`.

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
