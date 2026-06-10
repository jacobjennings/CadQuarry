            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v0.6.0 |
            | Seed             | 1234 |
            | Total parts      | 1000 |
            | Generated        | 2026-06-10 |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            | plate        |    141 |  14.1% |
| revolved     |    130 |  13.0% |
| bracket      |    121 |  12.1% |
| block        |     98 |   9.8% |
| compound     |     94 |   9.4% |
| flanged      |     53 |   5.3% |
| threaded     |     50 |   5.0% |
| sketched     |     47 |   4.7% |
| gear         |     47 |   4.7% |
| lofted       |     44 |   4.4% |
| enclosure    |     43 |   4.3% |
| tapped       |     36 |   3.6% |
| profiled     |     34 |   3.4% |
| swept        |     33 |   3.3% |
| ribbed       |     29 |   2.9% |

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            | Tier 0 |    149 |  14.9% |
| Tier 1 |    489 |  48.9% |
| Tier 2 |    287 |  28.7% |
| Tier 3 |     75 |   7.5% |

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
