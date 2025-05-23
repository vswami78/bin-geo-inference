"""
Module for loading and filtering transaction data.
"""

import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

# Assuming config.py is in the same directory or Python path
from config import TXN_SCHEMA, TRANSACTIONS_TABLE_NAME, DELTA_LAKE_PATH

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_filtered(spark: SparkSession, month: str) -> DataFrame:
    """
    Loads transaction data for a specific month from a Delta table, filters it 
    based on predefined criteria, and returns the filtered DataFrame.

    Args:
        spark: Active SparkSession.
        month: String in 'YYYY-MM' format. Currently, this argument is for logging and
               potential future use in partition filtering, but the read path
               does not implement partitioning by month. It reads the entire table.

    Returns:
        A DataFrame containing the filtered transaction data.
    """
    logging.info(f"Starting to load and filter data for month: {month}")

    table_path = f"{DELTA_LAKE_PATH}/{TRANSACTIONS_TABLE_NAME}"
    logging.info(f"Attempting to read data from Delta table: {table_path}")

    raw_df: DataFrame
    try:
        # In a real scenario with a partitioned table:
        # raw_df = spark.read.format("delta").load(table_path).where(F.col("partition_month_col") == month)
        raw_df = spark.read.format("delta").load(table_path)
        initial_count = raw_df.count()
        logging.info(f"Initial number of rows read from {table_path} for month {month}: {initial_count}")
    except AnalysisException as e:
        # This handles cases where the Delta table doesn't exist (e.g., first run or in tests)
        logging.warning(f"Could not read Delta table at {table_path}: {e}. "
                        "Assuming this is acceptable (e.g., for testing or initial run) and "
                        "returning an empty DataFrame with the expected schema.")
        raw_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), TXN_SCHEMA)
        initial_count = 0 # Since we created an empty DataFrame

    # Apply filters:
    # 1. Keep only non-fraudulent transactions (fraud_label == False)
    filtered_df = raw_df.filter(F.col("fraud_label") == False)

    # 2. Keep only rows where billing_address_country is not null
    filtered_df = filtered_df.filter(F.col("billing_address_country").isNotNull())

    # 3. Keep only rows where the BIN length is between 6 and 8 digits inclusive
    # Ensure BIN column exists and is string type for F.length to work as expected.
    # TXN_SCHEMA defines BIN as StringType.
    filtered_df = filtered_df.filter(
        (F.length(F.col("BIN")) >= 6) & (F.length(F.col("BIN")) <= 8)
    )

    final_count = filtered_df.count()
    rows_removed = initial_count - final_count
    logging.info(f"Number of rows after filtering for month {month}: {final_count}")
    logging.info(f"Number of rows removed by filtering: {rows_removed}")

    return filtered_df

if __name__ == '__main__':
    # This section is for ad-hoc local testing and demonstration.
    # It requires a Spark environment and will attempt to create a dummy Delta table.
    
    local_spark = SparkSession.builder \
        .appName("LoadFilterLocalTest") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") # Ensure Delta is available
        .getOrCreate()

    # Ensure DELTA_LAKE_PATH and the table directory exist for the dummy table
    dummy_table_full_path = f"{DELTA_LAKE_PATH}/{TRANSACTIONS_TABLE_NAME}"
    import os
    os.makedirs(dummy_table_full_path, exist_ok=True)
    logging.info(f"Ensured directory exists: {dummy_table_full_path}")

    # Sample data matching TXN_SCHEMA (transaction_id, timestamp, BIN, billing_address_country, fraud_label)
    from datetime import datetime
    sample_data = [
        ("tx_valid", datetime(2023, 1, 1, 10, 0, 0), "123456", "US", False),      # Valid
        ("tx_fraud", datetime(2023, 1, 1, 11, 0, 0), "654321", "CA", True),       # Fraud
        ("tx_null_country", datetime(2023, 1, 1, 12, 0, 0), "112233", None, False), # Null country
        ("tx_bin_short", datetime(2023, 1, 1, 13, 0, 0), "12345", "GB", False),    # BIN too short
        ("tx_bin_long", datetime(2023, 1, 1, 14, 0, 0), "123456789", "DE", False), # BIN too long
        ("tx_valid_7digit", datetime(2023, 1, 1, 15, 0, 0), "7654321", "FR", False), # Valid 7-digit BIN
        ("tx_valid_8digit", datetime(2023, 1, 1, 16, 0, 0), "87654321", "MX", False)  # Valid 8-digit BIN
    ]

    try:
        logging.info(f"Creating dummy DataFrame with schema: {TXN_SCHEMA.simpleString()}")
        source_df = local_spark.createDataFrame(sample_data, schema=TXN_SCHEMA)
        
        logging.info(f"Writing dummy data to Delta table: {dummy_table_full_path}")
        source_df.write.format("delta").mode("overwrite").save(dummy_table_full_path)
        logging.info("Dummy Delta table created/overwritten successfully.")

        # Test loading and filtering
        logging.info("Calling load_filtered function for month '2023-01'...")
        filtered_df_test = load_filtered(local_spark, "2023-01")
        
        logging.info("Filtered data sample from local test:")
        filtered_df_test.show(truncate=False)
        
        expected_count = 3 # tx_valid, tx_valid_7digit, tx_valid_8digit
        actual_count = filtered_df_test.count()
        logging.info(f"Count of filtered transactions in local test: {actual_count} (Expected: {expected_count})")
        
        if actual_count == expected_count:
            logging.info("Local test count matches expected count.")
        else:
            logging.error(f"Local test count mismatch: Got {actual_count}, expected {expected_count}")

    except Exception as e:
        logging.error(f"Error in local test run: {e}", exc_info=True)
    finally:
        # Optional: Clean up the dummy table path after test.
        # For robust cleanup, consider using tempfile module for paths.
        # import shutil
        # try:
        #     shutil.rmtree(dummy_table_full_path)
        #     logging.info(f"Cleaned up dummy Delta table at {dummy_table_full_path}")
        #     # Attempt to remove base path if it's empty and was created by this script context
        #     if DELTA_LAKE_PATH not in dummy_table_full_path and os.path.exists(DELTA_LAKE_PATH) and not os.listdir(DELTA_LAKE_PATH):
        #          os.rmdir(DELTA_LAKE_PATH)
        # except OSError as oe:
        #     logging.warning(f"Could not clean up all dummy data directories: {oe}")
            
        local_spark.stop()
        logging.info("Local Spark session stopped.")
