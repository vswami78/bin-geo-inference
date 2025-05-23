### `todo.md`: BIN Geolocation Pipeline Tasks

**[X] Unit 0: Define Project Config & Schema**
- [X] Create `config.py`
  - Constants: `MIN_TXNS`, `COVERAGE_PCT`, delta paths, table names
  - Spark `StructType` for transaction input

**[X] Unit 1: Load & Filter Transaction Data**
- [X] Implement `load_filtered(month)` in `load_filter.py`
  - Filter non-fraud, non-null billing country, valid BIN
- [X] Write unit tests for filtering logic

**[X] Unit 2: Aggregate Country Counts Per BIN**
- [X] Create `country_counts(df)` in `aggregate.py`
  - Output: `BIN`, `country`, `txn_count`, `bin_total`

**[X] Unit 3: Select Countries Covering ≥95% Usage**
- [X] Implement `select_top_countries(df)` in `coverage.py`
  - Cumulative % logic per BIN
- [X] Write unit tests for correct threshold coverage

**[X] Unit 4: Filter Low-Support BINs**
- [X] Add `filter_supported_bins(df)` to `coverage.py`
  - Drop rows below `MIN_TXNS`
- [X] Test BIN exclusion logic

**[X] Unit 5: Compare With Vendor Mapping**
- [X] Load vendor BIN-country mapping (Note: Assumed loaded and passed as DataFrame to `flag_discrepancies`)
- [X] Implement `flag_discrepancies(inferred_df, vendor_df)` in `compare_vendor.py`
  - Flag where top inferred country ≠ vendor's
- [X] Unit test with dummy data

**[X] Unit 6: Write to Monthly Delta Table**
- [X] Implement `write_delta(df, month)` in `write_output.py`
  - Append/merge with partition by `snapshot_month`
- [X] Validate output format and schema (Note: Validation done via unit tests reading back data and checking schema)

**[X] Unit 7: Build Pipeline Orchestration Script**
- [X] Create `bin_geo_pipeline.py` or Databricks notebook
  - Run steps in sequence with logging & error capture

**[X] Unit 8: Add Data Quality & Regression Tests**
- [X] Create `tests/test_regression.py`
  - Validate schema, nulls, valid countries, no duplicate BIN-country
