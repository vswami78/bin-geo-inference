"""
Main orchestration script for the BIN Geolocation Inference Pipeline.
"""

import argparse
import logging
import os
import shutil
import tempfile
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType

# Import pipeline functions from other modules
from load_filter import load_filtered
from aggregate import country_counts
from coverage import select_top_countries, filter_supported_bins
from compare_vendor import flag_discrepancies
from write_output import write_delta

# Import configuration constants
# We import 'config' as a module to allow monkeypatching its attributes during testing
# or for temporary overrides in this script.
import config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_spark_session(app_name="BinGeoPipeline") -> SparkSession:
    """Initializes and returns a SparkSession."""
    logging.info(f"Initializing SparkSession with app name: {app_name}")
    spark_builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
    
    # Check if running in a CI or test environment where local master is preferred
    if os.environ.get("SPARK_MASTER_URL"):
        spark_builder = spark_builder.master(os.environ["SPARK_MASTER_URL"])
    else:
        spark_builder = spark_builder.master("local[*]") # Default to local

    spark = spark_builder.getOrCreate()
    logging.info("SparkSession initialized successfully.")
    # Log Spark version and other relevant info if needed
    # logging.info(f"Spark version: {spark.version}")
    return spark

def setup_dummy_transactions_table(spark: SparkSession, month_to_simulate_data_for: str):
    """
    Creates a dummy 'transactions' Delta table with sample data for testing the pipeline.
    Writes to config.DELTA_LAKE_PATH / config.TRANSACTIONS_TABLE_NAME.
    """
    dummy_table_path = os.path.join(config.DELTA_LAKE_PATH, config.TRANSACTIONS_TABLE_NAME)
    logging.info(f"Setting up dummy transactions table at: {dummy_table_path}")

    # Sample data based on config.TXN_SCHEMA
    # Fields: transaction_id, timestamp, BIN, billing_address_country, fraud_label
    sample_data = [
        ("dummy_tx_001", datetime.now(), "450000", "US", False), ("dummy_tx_002", datetime.now(), "450000", "US", False),
        ("dummy_tx_003", datetime.now(), "450000", "CA", False), ("dummy_tx_004", datetime.now(), "510000", "GB", False),
        ("dummy_tx_005", datetime.now(), "510000", "GB", False), ("dummy_tx_006", datetime.now(), "4500011", "DE", False),
        ("dummy_tx_007", datetime.now(), "4500011", "DE", False),("dummy_tx_008", datetime.now(), "51000022", "FR", False),
        ("dummy_tx_009", datetime.now(), "51000022", "FR", False),("dummy_tx_010", datetime.now(), "51000022", "US", False),
        ("dummy_tx_f01", datetime.now(), "450000", "US", True), ("dummy_tx_f02", datetime.now(), "45000", "CA", False),
        ("dummy_tx_f03", datetime.now(), "450000001", "GB", False), ("dummy_tx_f04", datetime.now(), "510000", None, False),
    ]
    
    # Add more data for a specific BIN to ensure it passes MIN_TXNS
    # config.MIN_TXNS is used here (default 100 in config.py)
    for i in range(config.MIN_TXNS + 5): 
        sample_data.append((f"dummy_tx_main_bin_{i:03d}", datetime.now(), "123456", "US", False))
        if i % 3 == 0:
             sample_data.append((f"dummy_tx_main_bin_alt_{i:03d}", datetime.now(), "123456", "CA", False))

    try:
        dummy_df = spark.createDataFrame(sample_data, schema=config.TXN_SCHEMA)
        logging.info(f"Writing {dummy_df.count()} rows of dummy transaction data to {dummy_table_path}...")
        
        # Ensure the parent directory of the dummy table exists (config.DELTA_LAKE_PATH)
        os.makedirs(config.DELTA_LAKE_PATH, exist_ok=True)
        
        dummy_df.write.format("delta").mode("overwrite").save(dummy_table_path)
        logging.info(f"Successfully wrote dummy transactions to {dummy_table_path}")
    except Exception as e:
        logging.error(f"Failed to create dummy transactions table at {dummy_table_path}: {e}", exc_info=True)
        raise

def load_vendor_data(spark: SparkSession, table_name: str) -> DataFrame:
    """
    Placeholder for loading vendor data. Returns a dummy DataFrame.
    `table_name` from config is logged but not used for dummy data generation.
    """
    logging.warning(f"load_vendor_data: Using DUMMY vendor data. Configured table name '{table_name}' is NOT read.")
    
    vendor_schema = StructType([
        StructField("BIN", StringType(), True),
        StructField("country", StringType(), True) 
    ])
    dummy_vendor_data = [
        ("450000", "US"), ("510000", "CA"), ("123456", "US"), ("999999", "FR"),
        ("4500011", "DE"), ("51000022", "FR"),
    ]
    vendor_df = spark.createDataFrame(dummy_vendor_data, schema=vendor_schema)
    logging.info(f"load_vendor_data: Created dummy vendor DataFrame with {vendor_df.count()} rows.")
    return vendor_df

def main():
    """Main function to orchestrate the BIN geolocation pipeline."""
    parser = argparse.ArgumentParser(description="BIN Geolocation Inference Pipeline")
    parser.add_argument("--month", type=str, required=True, help="Month to process in YYYY-MM format.")
    parser.add_argument(
        "--use-dummy-source-data", action="store_true",
        help="If set, a dummy transactions Delta table will be created and used as source."
    )
    parser.add_argument(
        "--force-temp-delta-path", action="store_true",
        help="If set, forces config.DELTA_LAKE_PATH to a new temporary local directory for this run. "
             "Implies --use-dummy-source-data if the transactions table would be in this temp path."
    )
    args = parser.parse_args()
    month = args.month

    if not (len(month) == 7 and month[4] == '-'): # Basic format check
        logging.error(f"Invalid month format: '{month}'. Expected 'YYYY-MM'.")
        return

    logging.info(f"Starting BIN Geolocation Pipeline for month: {month}")

    spark = None
    original_config_delta_lake_path = config.DELTA_LAKE_PATH # Store original
    temp_delta_dir_for_run = None # Path to the temp dir if created

    try:
        spark = get_spark_session()

        # Override DELTA_LAKE_PATH with a temp dir if forced or if default is placeholder
        # This is critical for making the pipeline runnable in test/CI environments
        # without pre-existing infrastructure or accidental writes to production paths.
        is_placeholder_path = config.DELTA_LAKE_PATH == "/mnt/delta/bin_geolocation"
        if args.force_temp_delta_path or is_placeholder_path:
            if is_placeholder_path and not args.force_temp_delta_path:
                logging.warning(f"Default config.DELTA_LAKE_PATH ('{config.DELTA_LAKE_PATH}') "
                                "seems like a placeholder. Forcing use of a temporary directory for this run.")
            
            temp_delta_dir_for_run = tempfile.mkdtemp(prefix="bin_geo_pipeline_")
            logging.warning(f"Overriding config.DELTA_LAKE_PATH. "
                            f"Original: '{original_config_delta_lake_path}', New (temp): '{temp_delta_dir_for_run}'")
            config.DELTA_LAKE_PATH = temp_delta_dir_for_run
            
            # If using a temp path for DELTA_LAKE_PATH, it implies any source tables within it
            # (like the main transactions table) won't exist, so dummy source data is needed.
            if not args.use_dummy_source_data:
                logging.info("Using temporary Delta Lake path, so enabling --use-dummy-source-data.")
                args.use_dummy_source_data = True

        if args.use_dummy_source_data:
            logging.info("Setting up dummy source transactions table as requested/implied...")
            setup_dummy_transactions_table(spark, month)

        # --- Pipeline Steps ---
        current_df = None

        logging.info("Step 1: Load and Filter Transactions")
        current_df = load_filtered(spark, month)
        logging.info(f"Step 1 completed. Rows after filtering: {current_df.count()}")
        # current_df.show(5, truncate=False)

        logging.info("Step 2: Aggregate Country Counts")
        current_df = country_counts(current_df)
        logging.info(f"Step 2 completed. Rows after aggregation: {current_df.count()}")
        # current_df.show(5, truncate=False)

        logging.info("Step 3: Select Top Countries by Coverage")
        current_df = select_top_countries(current_df)
        logging.info(f"Step 3 completed. Rows after coverage selection: {current_df.count()}")
        # current_df.show(5, truncate=False)

        logging.info("Step 4: Filter Supported BINs (by MIN_TXNS)")
        current_df = filter_supported_bins(current_df)
        logging.info(f"Step 4 completed. Rows after MIN_TXNS filtering: {current_df.count()}")
        # current_df.show(5, truncate=False)
        
        logging.info("Step 5: Load Vendor Data")
        vendor_df = load_vendor_data(spark, config.VENDOR_MAPPING_TABLE_NAME)
        logging.info(f"Step 5 completed. Vendor data rows: {vendor_df.count()}")
        # vendor_df.show(5, truncate=False)

        logging.info("Step 6: Flag Discrepancies")
        current_df = flag_discrepancies(current_df, vendor_df)
        logging.info(f"Step 6 completed. Rows after flagging discrepancies: {current_df.count()}")
        # current_df.show(5, truncate=False)

        logging.info("Step 7: Write Output to Delta Table")
        write_delta(spark, current_df, month)
        logging.info(f"Step 7 completed. Output written for month {month}.")

        logging.info(f"BIN Geolocation Pipeline COMPLETED successfully for month: {month}!")

    except Exception as e:
        logging.error(f"Pipeline execution FAILED for month {month}: {e}", exc_info=True)
    finally:
        if spark:
            logging.info("Stopping SparkSession.")
            spark.stop()
        
        # Restore original DELTA_LAKE_PATH in the config module if it was changed
        if temp_delta_dir_for_run: # This implies config.DELTA_LAKE_PATH was monkeypatched
            config.DELTA_LAKE_PATH = original_config_delta_lake_path 
            logging.info(f"Restored config.DELTA_LAKE_PATH to: {config.DELTA_LAKE_PATH}")
            if os.path.exists(temp_delta_dir_for_run):
                logging.info(f"Cleaning up temporary Delta directory: {temp_delta_dir_for_run}")
                try:
                    shutil.rmtree(temp_delta_dir_for_run)
                    logging.info(f"Successfully removed temporary directory: {temp_delta_dir_for_run}")
                except Exception as e_rm:
                    logging.error(f"Failed to remove temporary directory {temp_delta_dir_for_run}: {e_rm}")
        elif original_config_delta_lake_path != config.DELTA_LAKE_PATH:
            # This case might occur if DELTA_LAKE_PATH was changed by some other means
            # or if the script exits unexpectedly before restoration.
            logging.warning(f"config.DELTA_LAKE_PATH ('{config.DELTA_LAKE_PATH}') "
                            f"differs from original ('{original_config_delta_lake_path}') "
                            "but no temporary directory was tracked by the pipeline's main().")


if __name__ == "__main__":
    main()
```
