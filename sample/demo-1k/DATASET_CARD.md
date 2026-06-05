            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v0.3.0 |
            | Seed             | 1234 |
            | Total parts      | 1000 |
            | Generated        | 2026-06-05 |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            | plate        |    211 |  21.1% |
| revolved     |    196 |  19.6% |
| bracket      |    173 |  17.3% |
| block        |    153 |  15.3% |
| flanged      |     78 |   7.8% |
| enclosure    |     66 |   6.6% |
| ribbed       |     66 |   6.6% |
| profiled     |     57 |   5.7% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    123 |  12.3% |
| Tier 1 |    486 |  48.6% |
| Tier 2 |    306 |  30.6% |
| Tier 3 |     85 |   8.5% |

            ## Reproducibility

            Re-generate this exact corpus:

            ```bash
            cadquarry generate --seed 1234 --count 1000 \
                --out <output_dir> --config configs/default.toml
            ```

            ## License

            Generated data: **CC0-1.0** (public domain dedication).
            Generator source: **Apache-2.0**.
            See DATA_LICENSE and LICENSE in the repository root.
