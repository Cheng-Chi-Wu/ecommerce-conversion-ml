# Dataset provenance and usage

Source: [Retailrocket recommender system dataset on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
Dataset provider: Retailrocket; the listing is maintained by Roman Zykov.
Listing checked on October 1, 2026. The listing displays **CC BY-NC-SA 4.0**:
[license text and terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The dataset was collected by its provider, not by this project's author.

The original files are `events.csv`, `item_properties_part1.csv`,
`item_properties_part2.csv`, and `category_tree.csv`. Obtain them from the source,
retain its terms, and put them in `data/raw/`. Raw files, row-level derivatives,
SQLite databases and trained models are excluded from Git.

Observed item properties form a timestamped change log. Category IDs and
availability have readable meanings; other property values are anonymized.
This project does not infer prices, revenue, product names or user demographics
from hashed fields. Category assignments are joined backward in time.

`reports/data_quality.json` records local file sizes, SHA-256 fingerprints,
observed columns, counts and dates. The source files were found in
`<LOCAL_DATA_DIR>` and placed in `data/raw/`. Filenames alone do not prove a dataset version; hashes
identify the exact local inputs used for this run.

Original code is provided under the MIT license in `LICENSE`. Dataset-derived
tables, figures and analyses in this repository retain attribution to
Retailrocket and are shared under CC BY-NC-SA 4.0. The MIT license does not
override dataset terms. No permission for commercial use of the data is implied.
