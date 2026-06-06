            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v0.5.0 |
            | Seed             | 1234 |
            | Total parts      | 1000 |
            | Generated        | 2026-06-05 |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            | bracket      |    162 |  16.2% |
| revolved     |    162 |  16.2% |
| plate        |    141 |  14.1% |
| block        |    110 |  11.0% |
| compound     |    109 |  10.9% |
| flanged      |     66 |   6.6% |
| threaded     |     66 |   6.6% |
| gear         |     54 |   5.4% |
| enclosure    |     50 |   5.0% |
| ribbed       |     49 |   4.9% |
| profiled     |     31 |   3.1% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    119 |  11.9% |
| Tier 1 |    498 |  49.8% |
| Tier 2 |    298 |  29.8% |
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
