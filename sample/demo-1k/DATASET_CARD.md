            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v0.4.0 |
            | Seed             | 1234 |
            | Total parts      | 1000 |
            | Generated        | 2026-06-05 |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            | revolved     |    193 |  19.3% |
| bracket      |    165 |  16.5% |
| plate        |    156 |  15.6% |
| compound     |    120 |  12.0% |
| block        |    116 |  11.6% |
| flanged      |     78 |   7.8% |
| ribbed       |     61 |   6.1% |
| enclosure    |     60 |   6.0% |
| profiled     |     51 |   5.1% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    110 |  11.0% |
| Tier 1 |    495 |  49.5% |
| Tier 2 |    313 |  31.3% |
| Tier 3 |     82 |   8.2% |

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
