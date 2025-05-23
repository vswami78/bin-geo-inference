"""
Module for aggregating transaction data to count occurrences per BIN and country.
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected output schema for country_counts function
# This helps in creating an empty DataFrame with schema if input is empty.
COUNTRY_COUNTS_SCHEMA = StructType([
    StructField("BIN", StringType(), True),
    StructField("country", StringType(), True),
    StructField("txn_count", LongType(), False), # count is typically LongType
    StructField("bin_total", LongType(), False)  # sum of counts is typically LongType
])

def country_counts(df: DataFrame) -> DataFrame:
    """
    Aggregates transaction data to count occurrences for each BIN-country pair
    and calculate the total transactions for each BIN.

    Args:
        df: Input DataFrame, expected to be the output from load_filtered,
            containing at least "BIN" and "billing_address_country" columns.

    Returns:
        A DataFrame with columns: "BIN", "country", "txn_count", "bin_total".
        Returns an empty DataFrame with the correct schema if the input is empty.
    """
    input_count = df.count()
    logging.info(f"Input DataFrame row count to country_counts: {input_count}")

    if input_count == 0:
        logging.info("Input DataFrame is empty. Returning an empty DataFrame with the expected schema.")
        # Need SparkSession to create an empty DataFrame.
        # This function doesn't take SparkSession as an arg.
        # One common pattern is to get it from the DataFrame itself.
        if df.sparkSession is None:
            # This case should ideally not happen if df is a valid DataFrame.
            # Fallback or raise error if no SparkSession is available.
            # For now, let's assume df.sparkSession will be available.
            # If not, creating an empty DF would need a global SparkSession or passing it.
            # For this exercise, we'll assume that operations on an empty df will yield an empty df with schema,
            # or tests will provide a spark session to create one.
            # The schema definition above (COUNTRY_COUNTS_SCHEMA) is for tests or if Spark is available.
            # If df is an empty DataFrame *with schema* (e.g. from createDataFrame([], schema)),
            # the transformations below will correctly produce an empty DataFrame with the new schema.
             spark = SparkSession.getActiveSession()
             if spark:
                 return spark.createDataFrame(spark.sparkContext.emptyRDD(), COUNTRY_COUNTS_SCHEMA)
             else:
                 # This is problematic; a function transforming DataFrames usually doesn't create a SparkSession.
                 # For now, we rely on Spark to handle empty input gracefully.
                 # The check for input_count == 0 is more for logging and early exit if possible.
                 logging.warning("SparkSession not available to create empty DataFrame with schema. "
                                 "Relying on transformations to produce correct empty schema.")


    # Step 1: Count transactions for each unique (BIN, billing_address_country) pair.
    # Alias billing_address_country to country.
    country_specific_counts_df = df.groupBy("BIN", F.col("billing_address_country").alias("country")) \
                                   .agg(F.count("*").alias("txn_count"))

    # Step 2: Calculate the total number of transactions for each BIN across all countries.
    window_spec_bin_total = Window.partitionBy("BIN")

    aggregated_df = country_specific_counts_df.withColumn(
        "bin_total",
        F.sum("txn_count").over(window_spec_bin_total)
    )

    # Select and order the final columns as per spec: BIN, country, txn_count, bin_total
    # Order by BIN and then by txn_count descending for consistent output (optional but good practice)
    final_df = aggregated_df.select("BIN", "country", "txn_count", "bin_total") \
                            .orderBy(F.col("BIN"), F.desc("txn_count"))


    output_count = final_df.count()
    logging.info(f"Output DataFrame row count from country_counts: {output_count}")

    return final_df

if __name__ == '__main__':
    # Example of how to run this locally (requires Spark setup and sample data)
    from pyspark.sql import SparkSession
    # Assuming config.py and its TXN_SCHEMA are available for creating sample data
    # If config.py is not in PYTHONPATH, this import might fail.
    # For ad-hoc test, we can define a simple schema here if needed.
    try:
        from config import TXN_SCHEMA
    except ImportError:
        # Define a fallback schema if config.py is not found (e.g. running file directly)
        from pyspark.sql.types import StructType, StructField, StringType, BooleanType, TimestampType
        TXN_SCHEMA = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("BIN", StringType(), True),
            StructField("billing_address_country", StringType(), True),
            StructField("fraud_label", BooleanType(), True)
        ])
        logging.warning("config.TXN_SCHEMA not found, using a fallback schema for __main__ example.")


    local_spark = SparkSession.builder \
        .appName("AggregateLocalTest") \
        .master("local[*]") \
        .getOrCreate()

    from datetime import datetime
    sample_data_for_agg = [
        ("tx1", datetime(2023,1,1,10,0,0), "123456", "US", False),
        ("tx2", datetime(2023,1,1,11,0,0), "123456", "US", False),
        ("tx3", datetime(2023,1,1,12,0,0), "123456", "CA", False),
        ("tx4", datetime(2023,1,1,13,0,0), "654321", "GB", False),
        ("tx5", datetime(2023,1,1,14,0,0), "654321", "GB", False),
        ("tx6", datetime(2023,1,1,15,0,0), "654321", "GB", False),
        ("tx7", datetime(2023,1,1,16,0,0), "654321", "FR", False),
        ("tx8", datetime(2023,1,1,17,0,0), "111111", "DE", False),
        ("tx9", datetime(2023,1,1,18,0,0), "222222", "US", False),
        ("tx10", datetime(2023,1,1,19,0,0), "222222", "US", False),
        ("tx11", datetime(2023,1,1,20,0,0), "222222", "CA", False),
        ("tx12", datetime(2023,1,1,21,0,0), "222222", "MX", False),
        ("tx13", datetime(2023,1,1,22,0,0), "222222", "MX", False),
        ("tx14", datetime(2023,1,1,23,0,0), "222222", "MX", False),
    ]
    
    input_df = local_spark.createDataFrame(sample_data_for_agg, schema=TXN_SCHEMA)
    logging.info("Sample input DataFrame for local testing (from __main__):")
    input_df.show(truncate=False)

    aggregated_results_df = country_counts(input_df)

    logging.info("Aggregated results from local testing (from __main__):")
    # Expected order is BIN ascending, then txn_count descending
    aggregated_results_df.show(truncate=False)
    
    # Validate some results (example)
    bin_222222_data = aggregated_results_df.filter(F.col("BIN") == "222222").orderBy(F.desc("txn_count"))
    bin_222222_data.show()
    first_row_bin_222222 = bin_222222_data.first()
    if first_row_bin_222222:
        assert first_row_bin_222222["country"] == "MX"
        assert first_row_bin_222222["txn_count"] == 3
        assert first_row_bin_222222["bin_total"] == 6
        logging.info("Local test for BIN 222222 (MX) passed basic validation.")

    local_spark.stop()
    logging.info("Local Spark session stopped (from __main__).")
