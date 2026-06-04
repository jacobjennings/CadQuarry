            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v0.1.0 |
            | Seed             | 1234 |
            | Total parts      | 1000 |
            | Generated        | 2026-06-04 |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            | revolved     |    211 |  21.1% |
| bracket      |    188 |  18.8% |
| plate        |    169 |  16.9% |
| block        |    159 |  15.9% |
| flanged      |     81 |   8.1% |
| ribbed       |     71 |   7.1% |
| enclosure    |     70 |   7.0% |
| profiled     |     51 |   5.1% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    120 |  12.0% |
| Tier 1 |    487 |  48.7% |
| Tier 2 |    313 |  31.3% |
| Tier 3 |     80 |   8.0% |

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
