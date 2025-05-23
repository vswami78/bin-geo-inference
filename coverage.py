"""
Module for selecting top countries based on transaction coverage percentage
and filtering BINs based on minimum transaction support.
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

# Assuming config.py is in the same directory or Python path
try:
    from config import COVERAGE_PCT, MIN_TXNS
except ImportError:
    logging.warning("Could not import COVERAGE_PCT or MIN_TXNS from config. Using defaults.")
    COVERAGE_PCT = 0.95 # Default if config not found
    MIN_TXNS = 10       # Default if config not found, choose a sensible small value for tests

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected output schema for select_top_countries function
# This schema is also the input schema for filter_supported_bins
SELECT_TOP_COUNTRIES_SCHEMA = StructType([
    StructField("BIN", StringType(), True),
    StructField("country", StringType(), True),
    StructField("txn_count", LongType(), False), 
    StructField("bin_total", LongType(), False), 
    StructField("country_pct", DoubleType(), False), 
    StructField("cumulative_pct", DoubleType(), False)
])

def select_top_countries(df: DataFrame) -> DataFrame:
    """
    Selects top countries for each BIN based on cumulative transaction coverage
    percentage, as defined by COVERAGE_PCT.

    The input DataFrame is expected to have "BIN", "country", "txn_count",
    and "bin_total" columns (typically the output from aggregate.country_counts).

    The function calculates the percentage of transactions for each country within
    its BIN, then calculates a cumulative percentage (ordered by transaction count
    descending, then country ascending for ties). It filters rows to keep
    only those necessary to meet the COVERAGE_PCT.

    Args:
        df: Input DataFrame with BIN, country, txn_count, and bin_total.

    Returns:
        A DataFrame with columns: "BIN", "country", "txn_count", "bin_total",
        "country_pct", and "cumulative_pct", containing only the rows that
        meet the coverage criteria. Returns an empty DataFrame with the
        correct schema if the input is empty.
    """
    input_rows = df.count()
    logging.info(f"select_top_countries: Input DataFrame has {input_rows} rows.")

    if input_rows == 0:
        logging.info("select_top_countries: Input DataFrame is empty. Returning empty DataFrame with defined schema.")
        active_spark_session = SparkSession.getActiveSession()
        if active_spark_session:
             return active_spark_session.createDataFrame(active_spark_session.sparkContext.emptyRDD(), SELECT_TOP_COUNTRIES_SCHEMA)
        else:
            logging.error("select_top_countries: No active SparkSession to create empty DataFrame.")
            raise RuntimeError("SparkSession not found for creating an empty DataFrame in select_top_countries.")

    df_with_pct = df.withColumn("country_pct", F.col("txn_count") / F.col("bin_total"))
    window_spec = Window.partitionBy("BIN") \
                        .orderBy(F.desc("txn_count"), F.asc("country"))
    df_with_cumulative_pct = df_with_pct.withColumn("cumulative_pct", F.sum("country_pct").over(window_spec))
    
    filtered_df = df_with_cumulative_pct.filter(
        (F.col("cumulative_pct") - F.col("country_pct")) < COVERAGE_PCT
    )

    result_df = filtered_df.select(
        "BIN", "country", "txn_count", "bin_total", "country_pct", "cumulative_pct"
    ).orderBy(F.col("BIN"), F.desc("txn_count"), F.asc("country"))

    final_rows = result_df.count()
    logging.info(f"select_top_countries: Output DataFrame has {final_rows} rows after coverage filter.")
    logging.info(f"select_top_countries: {input_rows - final_rows} rows were filtered out.")
    return result_df

def filter_supported_bins(df: DataFrame) -> DataFrame:
    """
    Filters the DataFrame to keep only rows where the BIN's total transaction count
    (`bin_total`) is greater than or equal to `MIN_TXNS`.

    The input DataFrame is expected to be the output of `select_top_countries`,
    conforming to `SELECT_TOP_COUNTRIES_SCHEMA`.

    Args:
        df: Input DataFrame with at least "BIN" and "bin_total" columns.

    Returns:
        A DataFrame containing only rows for BINs that meet the `MIN_TXNS` threshold.
        The schema of the output DataFrame is identical to the input.
    """
    input_rows = df.count()
    initial_unique_bins = df.select("BIN").distinct().count()
    logging.info(f"filter_supported_bins: Input DataFrame has {input_rows} rows and {initial_unique_bins} unique BINs.")
    logging.info(f"filter_supported_bins: Using MIN_TXNS threshold: {MIN_TXNS}")

    if input_rows == 0:
        logging.info("filter_supported_bins: Input DataFrame is empty. Returning empty DataFrame.")
        # Schema is already SELECT_TOP_COUNTRIES_SCHEMA, so returning df (empty) is fine.
        return df

    # Filter rows where bin_total is greater than or equal to MIN_TXNS
    filtered_df = df.filter(F.col("bin_total") >= MIN_TXNS)

    output_rows = filtered_df.count()
    final_unique_bins = filtered_df.select("BIN").distinct().count()
    bins_removed_count = initial_unique_bins - final_unique_bins

    logging.info(f"filter_supported_bins: Output DataFrame has {output_rows} rows and {final_unique_bins} unique BINs.")
    logging.info(f"filter_supported_bins: {input_rows - output_rows} rows were removed.")
    logging.info(f"filter_supported_bins: {bins_removed_count} unique BINs were removed due to low support.")

    return filtered_df

if __name__ == '__main__':
    local_spark_session = SparkSession.builder \
        .appName("CoverageLocalRun") \
        .master("local[*]") \
        .getOrCreate()

    try:
        from aggregate import COUNTRY_COUNTS_SCHEMA as AGG_SCHEMA
    except ImportError:
        logging.warning("Could not import COUNTRY_COUNTS_SCHEMA from aggregate for __main__ example. Defining a fallback.")
        AGG_SCHEMA = StructType([
            StructField("BIN", StringType(), True),
            StructField("country", StringType(), True),
            StructField("txn_count", LongType(), False),
            StructField("bin_total", LongType(), False)
        ])
    
    # Sample data for select_top_countries
    sample_agg_data = [
        ("BIN1", "US", 60, 100), ("BIN1", "CA", 30, 100), ("BIN1", "MX", 5, 100), ("BIN1", "GB", 5, 100),
        ("BIN2", "DE", 90, 100), ("BIN2", "FR", 10, 100),
        ("BIN3", "JP", 8, MIN_TXNS -1 if MIN_TXNS > 1 else 8), # BIN3 total below MIN_TXNS (e.g. 8 vs 10 or 99 vs 100)
        ("BIN3", "CN", (MIN_TXNS -1 if MIN_TXNS > 1 else 8) - 8 , MIN_TXNS -1 if MIN_TXNS > 1 else 8) if MIN_TXNS -1 > 8 else None, # ensure txn_count <= bin_total
        ("BIN4", "IN", 60, 120), ("BIN4", "AU", 30, 120), ("BIN4", "HK", 30, 120), # BIN4 total above MIN_TXNS
        ("BIN5", "US", MIN_TXNS, MIN_TXNS), # BIN5 total exactly MIN_TXNS
    ]
    # Clean up None entry for BIN3 if it occurred
    sample_agg_data_cleaned = [row for row in sample_agg_data if row is not None]
    if any(r[0] == "BIN3" and r[2] <0 for r in sample_agg_data_cleaned): # data integrity check
        sample_agg_data_cleaned = [r for r in sample_agg_data_cleaned if not (r[0] == "BIN3" and r[2] <0)]
        if not any(r[0] == "BIN3" for r in sample_agg_data_cleaned) and (MIN_TXNS -1 if MIN_TXNS > 1 else 8) > 0 : # Add back if removed and total > 0
             sample_agg_data_cleaned.append(("BIN3", "JP", (MIN_TXNS -1 if MIN_TXNS > 1 else 8), (MIN_TXNS -1 if MIN_TXNS > 1 else 8)))


    source_df_agg = local_spark_session.createDataFrame(sample_agg_data_cleaned, schema=AGG_SCHEMA)
    source_df_agg_sorted = source_df_agg.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    
    logging.info(f"Coverage_LocalRun (__main__): Input for select_top_countries:")
    source_df_agg_sorted.show(truncate=False)

    logging.info(f"Coverage_LocalRun (__main__): Using COVERAGE_PCT = {COVERAGE_PCT}, MIN_TXNS = {MIN_TXNS}")
    
    selected_countries_df = select_top_countries(source_df_agg_sorted)
    logging.info("Coverage_LocalRun (__main__): Output of select_top_countries:")
    selected_countries_df.show(truncate=False)

    supported_bins_df = filter_supported_bins(selected_countries_df)
    logging.info("Coverage_LocalRun (__main__): Output of filter_supported_bins:")
    supported_bins_df.show(truncate=False)

    # Validation example for __main__
    # BIN3 should be filtered out if its bin_total was MIN_TXNS - 1
    # BIN5 should be kept as its bin_total is MIN_TXNS
    
    if (MIN_TXNS -1 if MIN_TXNS > 1 else 8) < MIN_TXNS :
        assert supported_bins_df.filter(F.col("BIN") == "BIN3").count() == 0, "BIN3 should be filtered out"
    
    assert supported_bins_df.filter(F.col("BIN") == "BIN5").count() > 0, "BIN5 should be kept"
    
    logging.info("Coverage_LocalRun (__main__): Ad-hoc tests passed.")
    local_spark_session.stop()
```
