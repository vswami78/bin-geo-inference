"""
Data quality and regression tests for the output Delta table.
"""
import pytest
import os
import shutil
import tempfile
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType

# Import project-specific items
import config # Will be monkeypatched by fixtures
try:
    from compare_vendor import FLAG_DISCREPANCIES_OUTPUT_SCHEMA
except ImportError:
    logging.warning("test_regression: Could not import FLAG_DISCREPANCIES_OUTPUT_SCHEMA from compare_vendor. Using fallback.")
    FLAG_DISCREPANCIES_OUTPUT_SCHEMA = StructType([
        StructField("BIN", StringType(), True),
        StructField("inferred_top_country", StringType(), True), # Should be non-null in valid output
        StructField("vendor_country", StringType(), True),       # Can be null
        StructField("is_discrepant", BooleanType(), False),     # Should be non-null
        StructField("bin_total", LongType(), True)              # Can be null if original data had issues
    ])


@pytest.fixture(scope="session")
def spark_session():
    """Session-scoped SparkSession fixture configured for Delta Lake."""
    spark = SparkSession.builder \
        .appName("RegressionTests") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def setup_test_delta_table_for_regression(spark_session, monkeypatch):
    """
    Sets up a temporary Delta table with controlled sample data for regression tests.
    This table simulates the output INFERRED_BIN_COUNTRY_MAP_TABLE_NAME.
    """
    temp_delta_root_dir = tempfile.mkdtemp(prefix="reg_test_delta_")
    test_output_table_name = "regression_output_table" 
    
    monkeypatch.setattr(config, 'DELTA_LAKE_PATH', temp_delta_root_dir)
    monkeypatch.setattr(config, 'INFERRED_BIN_COUNTRY_MAP_TABLE_NAME', test_output_table_name)
    
    table_path = os.path.join(temp_delta_root_dir, test_output_table_name)
    logging.info(f"Regression test setup: Using temporary Delta table path: {table_path}")

    # Data schema: BIN, inferred_top_country, vendor_country, is_discrepant, bin_total
    # Plus snapshot_month for partitioning.

    data_month1 = [
        ("450000", "US", "US", False, 1500L),       # Valid
        ("510000", "GB", "CA", True, 200L),        # Valid, Discrepancy
        ("123456", "DE", None, False, 100L),       # Valid, vendor_country is None
        ("VALIDC1", "FR", "FR", False, 75L),       # Valid country code
        ("BINLENOK", "JP", "JP", False, 90L),      # Valid BIN length (8)
        ("BINLEN6", "AU", "AU", False, 80L),       # Valid BIN length (6)
    ]
    
    data_month2 = [
        ("450000", "US", "US", False, 1600L),       # Valid, same BIN as month1
        ("789012", "MX", "MX", False, 300L),       # Valid
        # Data for specific DQ violations:
        ("BADCTRY1", "X", "X", False, 50L),        # Invalid country code length (1 char)
        ("BADCTRY2", "XYZ", "XYZ", False, 55L),    # Invalid country code length (3 chars)
        ("SHORTBIN", "CA", "CA", False, 60L),      # Invalid BIN length (too short: 8 chars) -> "SHORTBIN" is 8. Let's use "SHORT"
        ("LONGBIN", "BR", "BR", False, 70L),       # Invalid BIN length (too long: 7 chars) -> "LONGBIN" is 7. Let's use "LONGBINTOOLONG"
         # Row with null in a normally non-null field (for testing null check)
        (None, "US", "US", False, 100L),           # Null BIN
        ("NULCNTRY", None, "US", True, 110L),      # Null inferred_top_country (schema allows, but DQ test might not)
        ("NULDISC", "DE", "DE", None, 120L),       # Null is_discrepant (schema disallows, test this)
    ]
    
    # Corrected data for month2 based on test intentions
    data_month2_corrected = [
        ("450000", "US", "US", False, 1600L),
        ("789012", "MX", "MX", False, 300L),
        ("BADCTRY1", "X", "X", False, 50L),      # Invalid country code (1 char)
        ("BADCTRY2", "XYZ", "XYZ", False, 55L),  # Invalid country code (3 chars)
        ("SHORT", "CA", "CA", False, 60L),       # Invalid BIN length (5 chars)
        ("LONGBINTOOLONG", "BR", "BR", False, 70L), # Invalid BIN length (14 chars)
        (None, "US", "US", False, 100L),         # Null BIN
        ("NULCNTRY", None, "US", True, 110L),    # Null inferred_top_country
        # is_discrepant is non-nullable in schema, so can't directly test for its nullness via data.
        # Test will check if any nulls appear despite schema.
    ]


    df_m1 = spark_session.createDataFrame(data_month1, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA) \
                        .withColumn("snapshot_month", F.lit("2023-01"))
    
    df_m2 = spark_session.createDataFrame(data_month2_corrected, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA) \
                        .withColumn("snapshot_month", F.lit("2023-02"))

    try:
        df_m1.write.format("delta").mode("overwrite").partitionBy("snapshot_month").save(table_path)
        df_m2.write.format("delta").mode("overwrite").partitionBy("snapshot_month").option("partitionOverwriteMode", "dynamic").save(table_path)
        logging.info(f"Regression test setup: Test Delta table created and populated at {table_path}")
    except Exception as e:
        logging.error(f"Regression test setup: Failed to write test Delta table: {e}", exc_info=True)
        if os.path.exists(temp_delta_root_dir): shutil.rmtree(temp_delta_root_dir)
        raise

    yield spark_session, table_path

    logging.info(f"Regression test teardown: Removing temporary Delta directory: {temp_delta_root_dir}")
    if os.path.exists(temp_delta_root_dir):
        shutil.rmtree(temp_delta_root_dir)

# --- Regression Test Functions ---

def test_schema_validation(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path)
    
    # Add snapshot_month to expected schema for comparison as it's part of the table
    expected_schema_with_partition = FLAG_DISCREPANCIES_OUTPUT_SCHEMA.add("snapshot_month", StringType(), True)
    
    assert df.schema == expected_schema_with_partition, \
        f"Schema mismatch. Expected: {expected_schema_with_partition}, Got: {df.schema}"

def test_no_nulls_in_critical_columns(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path).filter("snapshot_month = '2023-01'") # Test on 'cleaner' data partition

    critical_cols = ["BIN", "inferred_top_country", "is_discrepant", "snapshot_month"]
    for col_name in critical_cols:
        assert df.filter(F.col(col_name).isNull()).count() == 0, f"Nulls found in critical column: {col_name}"

def test_bin_uniqueness_per_snapshot(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path)

    # Test data for month '2023-01' is designed to be unique per BIN
    # Test data for month '2023-02' has a null BIN, which group by will treat as a value.
    # Exclude rows where BIN is null for this uniqueness test, as nulls are not 'unique' in the typical sense.
    df_valid_bins = df.filter(F.col("BIN").isNotNull())
    
    bin_counts_per_snapshot = df_valid_bins.groupBy("snapshot_month", "BIN").count()
    duplicate_bins = bin_counts_per_snapshot.filter(F.col("count") > 1)
    
    assert duplicate_bins.count() == 0, f"Duplicate BINs found within snapshots: {duplicate_bins.collect()}"

def test_valid_country_codes(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path)

    # Test on rows where country codes are expected to be valid (e.g., '2023-01' partition)
    # and specifically exclude known bad data from '2023-02' used for testing this.
    df_valid_partition = df.filter("snapshot_month = '2023-01'")

    # Check inferred_top_country (should be non-null and 2 chars)
    assert df_valid_partition.filter(F.length(F.col("inferred_top_country")) != 2).count() == 0, \
        "Invalid inferred_top_country length found for valid partition."
    
    # Check vendor_country (if not null, should be 2 chars)
    assert df_valid_partition.filter(F.col("vendor_country").isNotNull() & (F.length(F.col("vendor_country")) != 2)).count() == 0, \
        "Invalid vendor_country length found for valid partition."

    # Now, check that our bad data in '2023-02' *does* violate this if we didn't filter
    df_bad_data_partition = df.filter("snapshot_month = '2023-02'")
    # BADCTRY1 has 'X' (len 1), BADCTRY2 has 'XYZ' (len 3)
    assert df_bad_data_partition.filter(
        (F.col("BIN") == "BADCTRY1") & (F.length(F.col("inferred_top_country")) != 2)
    ).count() == 1
    assert df_bad_data_partition.filter(
        (F.col("BIN") == "BADCTRY2") & (F.length(F.col("inferred_top_country")) != 2)
    ).count() == 1


def test_bin_length_validation(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path)

    # Test on rows where BINs are expected to be valid ('2023-01' partition)
    df_valid_partition = df.filter("snapshot_month = '2023-01'")
    invalid_bin_lengths_valid_partition = df_valid_partition.filter(
        (F.length(F.col("BIN")) < 6) | (F.length(F.col("BIN")) > 8)
    )
    assert invalid_bin_lengths_valid_partition.count() == 0, "Invalid BIN lengths found in '2023-01' (expected all valid)."

    # Test that our specific bad data in '2023-02' violates this
    df_bad_data_partition = df.filter("snapshot_month = '2023-02'")
    # SHORT has length 5, LONGBINTOOLONG has length 14
    assert df_bad_data_partition.filter(F.col("BIN") == "SHORT").count() == 1 # Check it exists
    assert df_bad_data_partition.filter(
        (F.col("BIN") == "SHORT") & ((F.length(F.col("BIN")) < 6) | (F.length(F.col("BIN")) > 8))
    ).count() == 1
    
    assert df_bad_data_partition.filter(F.col("BIN") == "LONGBINTOOLONG").count() == 1 # Check it exists
    assert df_bad_data_partition.filter(
        (F.col("BIN") == "LONGBINTOOLONG") & ((F.length(F.col("BIN")) < 6) | (F.length(F.col("BIN")) > 8))
    ).count() == 1
    

def test_snapshot_month_column_exists_and_populated(setup_test_delta_table_for_regression):
    spark, table_path = setup_test_delta_table_for_regression
    df = spark.read.format("delta").load(table_path)

    assert "snapshot_month" in df.columns, "snapshot_month column does not exist."
    
    # Check for a specific test month partition
    month_to_check = "2023-01"
    df_partition = df.filter(F.col("snapshot_month") == month_to_check)
    assert df_partition.count() > 0, f"No data found for snapshot_month = {month_to_check}"
    
    # All rows within this filtered DataFrame should have the correct snapshot_month value
    # (This is inherently true due to the filter, but a count check is a simple validation)
    mismatched_months = df_partition.filter(F.col("snapshot_month") != month_to_check).count()
    assert mismatched_months == 0, f"Rows found with incorrect snapshot_month value in partition {month_to_check}"

```
