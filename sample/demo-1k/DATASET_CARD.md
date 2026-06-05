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
            | revolved     |    213 |  21.3% |
| bracket      |    191 |  19.1% |
| plate        |    167 |  16.7% |
| block        |    152 |  15.2% |
| flanged      |     81 |   8.1% |
| ribbed       |     73 |   7.3% |
| enclosure    |     72 |   7.2% |
| profiled     |     51 |   5.1% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    120 |  12.0% |
| Tier 1 |    495 |  49.5% |
| Tier 2 |    311 |  31.1% |
| Tier 3 |     74 |   7.4% |

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
