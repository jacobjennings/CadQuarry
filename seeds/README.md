# Seed list

[`seeds.toml`](seeds.toml) is the single, consolidated list of **canonical
reproducible corpora**. A corpus is fully determined by the generator version
plus a seed, count, and config — so this list (with the generator) *is* the
deliverable; any published dataset is just a convenience artifact you can
regenerate at will.

It pins exactly **two seeds**, and within each only the part **count** changes
between entries:

- **`[meta].sample_seed`** — the canonical sample corpora under `[[corpus]]`:
  `smoke` (100 parts) and `demo-1k` (1000 parts, the corpus committed under
  [`sample/demo-1k/`](../sample/demo-1k/) and served on GitHub Pages). `smoke`
  is the first 100 parts of `demo-1k`.
- **`[publish].base_seed`** — the published Hugging Face size ladder under
  `[[publish.corpus]]`: `1k`…`100k`.

Generation is a deterministic prefix stream (each attempt is a pure function of
`(seed, index)` and accept/dedup runs in strict attempt order), so the first
*N* accepted parts of a run **are** the count-*N* corpus. Every size is a
`head -N` slice of the same base, and a base can be **extended** to larger sizes
later without redoing work (`cadquarry build --extend-to N`).

## Reproduce a corpus

```bash
# e.g. the "smoke" sample (sample_seed = 1234)
cadquarry generate --seed 1234 --count 100 \
    --config configs/default.toml --out datasets/smoke/

# confirm it re-executes and signatures hold
cadquarry verify datasets/smoke/
```

Same `generator_version` + same seed + same config ⇒ identical `manifest.jsonl`
and identical geometry signatures, bit-for-bit.

## Versioning

`[meta].generator_version` and `[meta].signature_version` pin the exact
generator and geometry-signature definition this list targets. Reproducing a
corpus requires matching `cadquarry.__version__` and `filter.SIGNATURE_VERSION`;
mismatches can change which parts count as duplicates (and therefore the
corpus).
