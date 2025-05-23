# prompt_plan.md

prompt_plan.md
Unit #	Objective	Expected Code Output	Key Test Cases	Prompt Instructions for Codegen LLM
0	Project constants & schema definitions
Set global parameters (e.g., MIN_TXNS, COVERAGE_PCT, table names) and common Spark schema for inputs/outputs.	config.py (Python module)	• All constants present and correctly typed
• Importable without error	“Create config.py defining constants below and a Spark StructType TXN_SCHEMA that matches the spec…” (list constants)
1	Load & filter source data
Read monthly slice of transactions table, keep non-fraud rows with non-null billing-country and valid BIN length (6–8).	load_filter.py with load_filtered(month) returning Spark DataFrame	• Filters drop fraud rows
• Null/invalid country rows removed
• BIN length check unit-tested	“Using Spark (PySpark), implement load_filtered(month) per config rules; include simple logging and unit tests with pytest + spark_session fixture.”
2	Aggregate countries per BIN
Count transactions by (BIN, country); compute total txns per BIN.	aggregate.py with country_counts(df) returning aggregated DF	• Aggregation columns correct
• Total count equals sum of country counts per BIN	“Write country_counts(df) that returns columns: BIN, country, txn_count, bin_total.”
3	Select dominant countries (≥ 95% coverage)	coverage.py with select_top_countries(df)	• For each BIN, cumulative pct ≥ 95%
• Country ordering by count desc	“Implement function to add cumulative_pct column, filter rows until coverage_pct in config reached.”
4	Enforce minimum-support BIN filter	Extension in coverage.py (filter_supported_bins)	• BINs with total < MIN_TXNS excluded	“Add helper that removes rows whose bin_total < MIN_TXNS; write unit test.”
5	Compare with vendor mapping
Left-join inferred mapping with vendor table; flag mismatches.	compare_vendor.py with flag_discrepancies(inferred_df, vendor_df)	• Discrepancy flag true when top inferred country ≠ vendor country	“Implement join & boolean is_discrepant; include test using small sample DFs.”
6	Write monthly delta snapshot	write_output.py with write_delta(df, month)	• Delta table exists & versioned
• Schema equals spec	“Write Spark code to merge/overwrite into delta path from config, adding snapshot_month partition.”
7	Orchestration notebook / job
Glue units 1-6 into single monthly run.	bin_geo_pipeline.py (or Databricks notebook)	• End-to-end run completes with sample month
• Output row count > 0	“Create driver script that imports previous modules, accepts --month arg (YYYY-MM), runs steps, catches & logs errors.”
8	Regression & data-quality test suite	tests/test_regression.py	• Validate schema, nulls, duplicate BIN-country rows	“Write pytest module to read latest delta snapshot and assert quality rules in spec.md.”

