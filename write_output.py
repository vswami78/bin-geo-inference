"""
Module for writing DataFrame to a Delta table with partitioning.
"""

import logging
import os # For __main__ example path creation

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Assuming config.py is in the same directory or Python path
try:
    from config import DELTA_LAKE_PATH, INFERRED_BIN_COUNTRY_MAP_TABLE_NAME
except ImportError:
    logging.warning("Could not import DELTA_LAKE_PATH or INFERRED_BIN_COUNTRY_MAP_TABLE_NAME from config. Using defaults for isolated run.")
    DELTA_LAKE_PATH = "/tmp/delta_lake_data_default" 
    INFERRED_BIN_COUNTRY_MAP_TABLE_NAME = "inferred_bin_country_map_default"

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_delta(spark: SparkSession, df: DataFrame, month: str):
    """
    Writes a DataFrame to a Delta table, partitioned by 'snapshot_month'.
    Uses dynamic partition overwrite mode.

    Args:
        spark: Active SparkSession.
        df: Input DataFrame to be written.
        month: String in 'YYYY-MM' format, used as the partition value.
    """
    input_rows = df.count()
    logging.info(f"write_delta: Attempting to write DataFrame with {input_rows} rows for month {month}.")

    if not isinstance(month, str) or len(month) != 7 or month[4] != '-':
        logging.error(f"write_delta: Invalid month format: '{month}'. Expected 'YYYY-MM'.")
        raise ValueError(f"Invalid month format: '{month}'. Expected 'YYYY-MM'.")

    # Add snapshot_month column for partitioning
    df_with_partition_col = df.withColumn("snapshot_month", F.lit(month))

    # Construct the full path to the Delta table
    # This is where the table data will be stored.
    table_path = f"{DELTA_LAKE_PATH}/{INFERRED_BIN_COUNTRY_MAP_TABLE_NAME}"
    logging.info(f"write_delta: Target Delta table path: {table_path}")

    original_partition_overwrite_mode = None
    try:
        # Store original mode and set to dynamic
        original_partition_overwrite_mode = spark.conf.get("spark.sql.sources.partitionOverwriteMode", "static") # Default is static
        logging.info(f"write_delta: Original spark.sql.sources.partitionOverwriteMode: {original_partition_overwrite_mode}")
        
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        logging.info("write_delta: Set spark.sql.sources.partitionOverwriteMode to 'dynamic'.")

        logging.info(f"write_delta: Writing to Delta table at {table_path} with partition snapshot_month={month}.")
        
        # Ensure parent directories for the table_path exist if they are part of DELTA_LAKE_PATH
        # This is more for local file system based Delta tables.
        # For cloud storage, this is usually not needed or handled differently.
        # For this exercise, we assume DELTA_LAKE_PATH itself exists.
        # The Delta writer will create the INFERRED_BIN_COUNTRY_MAP_TABLE_NAME directory if it doesn't exist.
        # os.makedirs(table_path, exist_ok=True) # Usually not needed for Spark write

        df_with_partition_col.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .partitionBy("snapshot_month") \
            .save(table_path)

        logging.info(f"write_delta: Successfully wrote {input_rows} rows to {table_path} for partition snapshot_month={month}.")

    except Exception as e:
        logging.error(f"write_delta: Failed to write DataFrame to Delta table at {table_path}. Error: {e}", exc_info=True)
        raise 
    finally:
        # Restore original partition overwrite mode
        if original_partition_overwrite_mode is not None: # Should always be true due to default in get
            spark.conf.set("spark.sql.sources.partitionOverwriteMode", original_partition_overwrite_mode)
            logging.info(f"write_delta: Restored spark.sql.sources.partitionOverwriteMode to '{original_partition_overwrite_mode}'.")


if __name__ == '__main__':
    import tempfile
    import shutil
    
    local_spark = SparkSession.builder \
        .appName("WriteDeltaLocalAdhocTest") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .getOrCreate()

    # Create a temporary directory for this ad-hoc test
    temp_adhoc_delta_dir = tempfile.mkdtemp(prefix="adhoc_delta_")
    
    # Store original config values to restore them later
    _original_config_delta_path = DELTA_LAKE_PATH
    _original_config_table_name = INFERRED_BIN_COUNTRY_MAP_TABLE_NAME
    
    # Override config values for this ad-hoc test
    DELTA_LAKE_PATH = temp_adhoc_delta_dir 
    INFERRED_BIN_COUNTRY_MAP_TABLE_NAME = "adhoc_test_table"
    
    adhoc_table_full_path = f"{DELTA_LAKE_PATH}/{INFERRED_BIN_COUNTRY_MAP_TABLE_NAME}"
    logging.info(f"Adhoc Test: Using temporary Delta path: {adhoc_table_full_path}")

    try:
        from compare_vendor import FLAG_DISCREPANCIES_OUTPUT_SCHEMA as AdhocSchema
    except ImportError:
        logging.warning("Adhoc Test: Could not import FLAG_DISCREPANCIES_OUTPUT_SCHEMA. Using fallback.")
        from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType
        AdhocSchema = StructType([
            StructField("BIN", StringType(), True), StructField("inferred_top_country", StringType(), True),
            StructField("vendor_country", StringType(), True), StructField("is_discrepant", BooleanType(), False),
            StructField("bin_total", LongType(), True)
        ])

    data_m1 = [("B1", "US", "CA", True, 100L), ("B2", "DE", "DE", False, 200L)]
    df_m1 = local_spark.createDataFrame(data_m1, schema=AdhocSchema)
    
    data_m2 = [("B1", "US", "US", False, 150L), ("B3", "FR", None, False, 50L)]
    df_m2 = local_spark.createDataFrame(data_m2, schema=AdhocSchema)

    data_m1_new = [("B1", "US", "CA", True, 120L), ("B4", "GB", "GB", False, 80L)] # B2 from m1 is gone
    df_m1_new = local_spark.createDataFrame(data_m1_new, schema=AdhocSchema)

    logging.info("Adhoc Test: Writing for 2023-01 (initial)...")
    write_delta(local_spark, df_m1, "2023-01")
    
    logging.info("Adhoc Test: Writing for 2023-02...")
    write_delta(local_spark, df_m2, "2023-02")

    logging.info("Adhoc Test: Reading full table after M1 and M2 writes...")
    full_df_1 = local_spark.read.format("delta").load(adhoc_table_full_path)
    full_df_1.show(truncate=False)
    assert full_df_1.count() == 4
    assert full_df_1.where("snapshot_month = '2023-01'").count() == 2
    assert full_df_1.where("snapshot_month = '2023-02'").count() == 2

    logging.info("Adhoc Test: Overwriting 2023-01 data...")
    write_delta(local_spark, df_m1_new, "2023-01")

    logging.info("Adhoc Test: Reading full table after M1 overwrite...")
    full_df_2 = local_spark.read.format("delta").load(adhoc_table_full_path)
    full_df_2.show(truncate=False)
    assert full_df_2.count() == 4 # 2 new for M1, 2 old for M2
    
    m1_content_after_overwrite = full_df_2.where("snapshot_month = '2023-01'").select("BIN").rdd.flatMap(lambda x: x).collect()
    assert "B2" not in m1_content_after_overwrite # B2 was in original M1, should be gone
    assert "B4" in m1_content_after_overwrite   # B4 is in new M1

    m2_content_after_overwrite = full_df_2.where("snapshot_month = '2023-02'").select("BIN").rdd.flatMap(lambda x: x).collect()
    assert "B3" in m2_content_after_overwrite   # B3 from M2 should still be there

    logging.info("Adhoc Test: All assertions passed.")

  
    # Test schema evolution
    from pyspark.sql.types import IntegerType
    evolved_schema = AdhocSchema.add(StructField("new_col_int", IntegerType(), True))
    data_m3_evolved = [("B5", "US", "US", False, 200L, 999)]
    df_m3_evolved = local_spark.createDataFrame(data_m3_evolved, schema=evolved_schema)

    logging.info("Adhoc Test: Writing for 2023-03 with evolved schema...")
    write_delta(local_spark, df_m3_evolved, "2023-03")
    
    logging.info("Adhoc Test: Reading full table after schema evolution...")
    full_df_3 = local_spark.read.format("delta").load(adhoc_table_full_path)
    full_df_3.show(truncate=False)
    assert "new_col_int" in full_df_3.columns
    assert full_df_3.where("snapshot_month = '2023-03' and new_col_int = 999").count() == 1
    # Check that older partitions have null for the new column
    assert full_df_3.where("snapshot_month = '2023-01' and new_col_int IS NULL").count() == 2


    logging.info("Adhoc Test: Schema evolution test passed.")
    logging.info("Adhoc Test: __main__ block finished.")
    
    
    
    
    except Exception as e:
        logging.error(f"Adhoc Test: FAILED with error: {e}", exc_info=True)
    finally:
        # Restore original config values
        DELTA_LAKE_PATH = _original_config_delta_path
        INFERRED_BIN_COUNTRY_MAP_TABLE_NAME = _original_config_table_name
        
        local_spark.stop()
        if os.path.exists(temp_adhoc_delta_dir):
            shutil.rmtree(temp_adhoc_delta_dir)
        logging.info(f"Adhoc Test: Cleaned up temp dir {temp_adhoc_delta_dir} and stopped Spark.")

```
