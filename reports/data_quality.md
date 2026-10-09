# Data quality audit

| file | rows | exact duplicates | missing values | UTC start | UTC end |
| --- | --- | --- | --- | --- | --- |
| events.csv | 2756101 | 460 | 2733644 | 2015-05-03 03:00:04.384000+00:00 | 2015-09-18 02:59:47.788000+00:00 |
| item_properties_part1.csv | 10999999 | 0 | 0 | 2015-05-10 03:00:00+00:00 | 2015-09-13 03:00:00+00:00 |
| item_properties_part2.csv | 9275903 | 0 | 0 | 2015-05-10 03:00:00+00:00 | 2015-09-13 03:00:00+00:00 |
| category_tree.csv | 1669 | 0 | 25 | — | — |

Exact event duplicates are removed from analytics and modeling. All property rows are retained for provenance; category history uses distinct updates. Missing transaction IDs on views/carts and missing parent IDs on root categories are structural.

Full column types, null counts, event distribution, SHA-256 fingerprints, temporal coverage and integrity checks are in [data_quality.json](data_quality.json).
