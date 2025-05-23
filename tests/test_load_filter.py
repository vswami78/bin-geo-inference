"""
Unit tests for the load_filter.py module.
"""
import pytest
import os
import shutil
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Assuming 'load_filter.py' is in the parent directory or accessible via PYTHONPATH
# For robust imports, ensure your project structure allows this.
# If load_filter.py is at the root and tests/ is a subdir, this should work if root is in PYTHONPATH.
from load_filter import load_filtered
from config import TXN_SCHEMA, DELTA_LAKE_PATH, TRANSACTIONS_TABLE_NAME

@pytest.fixture(scope="session")
def spark_session():
    """
    Pytest fixture to create a SparkSession for testing.
    Includes Delta Lake configurations.
    Manages a temporary path for the dummy Delta table.
    """
    temp_delta_path_base = os.path.join(DELTA_LAKE_PATH, "test_data")
    # Use a more specific sub-directory for this test session's table to avoid conflicts
    # and make cleanup easier if something goes wrong.
    # This will be DELTA_LAKE_PATH / test_data / TRANSACTIONS_TABLE_NAME
    # We will use the actual TRANSACTIONS_TABLE_NAME as the table name within temp_delta_path_base
    # So, load_filtered will read from DELTA_LAKE_PATH / TRANSACTIONS_TABLE_NAME
    # The fixture needs to ensure this path is managed.

    # We will override DELTA_LAKE_PATH for the test's scope to point to a temp dir
    # and then TRANSACTIONS_TABLE_NAME will be relative to that.
    
    # Create a temporary directory for Delta tables for this test session
    # This is safer than potentially polluting a real DELTA_LAKE_PATH
    # We'll use a generic temp path and make our config constants point there for the test.
    
    # A more robust way for tests is to use a temporary directory for DELTA_LAKE_PATH
    # and not rely on the configured one.
    # However, load_filtered directly imports DELTA_LAKE_PATH.
    # One option is to monkeypatch config.DELTA_LAKE_PATH for the test session.
    # Another is to make load_filtered accept DELTA_LAKE_PATH as an argument.
    # For now, we'll create the dummy table in the location load_filtered expects.
    
    test_specific_delta_lake_path = os.path.join(temp_delta_path_base, "load_filter_tests_temp_delta_root")
    dummy_table_full_path = os.path.join(test_specific_delta_lake_path, TRANSACTIONS_TABLE_NAME)

    if os.path.exists(test_specific_delta_lake_path):
        shutil.rmtree(test_specific_delta_lake_path)
    os.makedirs(dummy_table_full_path, exist_ok=True)

    spark = SparkSession.builder \
        .appName("LoadFilterTests") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()

    yield spark, test_specific_delta_lake_path, dummy_table_full_path # Provide path for test setup

    spark.stop()
    # Clean up the temporary Delta Lake path used for tests
    if os.path.exists(test_specific_delta_lake_path):
        shutil.rmtree(test_specific_delta_lake_path)


def _write_test_data_to_delta(spark: SparkSession, data: list, table_path: str):
    """Helper to write data to the dummy Delta table for a test."""
    if not data:
        # Create empty DF with schema if data is empty
        df = spark.createDataFrame(spark.sparkContext.emptyRDD(), TXN_SCHEMA)
    else:
        df = spark.createDataFrame(data, schema=TXN_SCHEMA)
    
    df.write.format("delta").mode("overwrite").save(table_path)


# Test cases
# Each test will prepare data, write it using the fixture's path, then call load_filtered.

def test_filter_fraudulent(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    test_data = [
        ("tx1", datetime(2023, 1, 1, 10, 0, 0), "123456", "US", True),  # Fraud
        ("tx2", datetime(2023, 1, 1, 11, 0, 0), "654321", "CA", False), # Valid
    ]
    _write_test_data_to_delta(spark, test_data, table_path)
    
    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 1
    assert filtered_df.filter(F.col("transaction_id") == "tx2").count() == 1


def test_filter_null_country(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    test_data = [
        ("tx1", datetime(2023, 1, 1, 10, 0, 0), "123456", None, False),   # Null country
        ("tx2", datetime(2023, 1, 1, 11, 0, 0), "654321", "CA", False),  # Valid
    ]
    _write_test_data_to_delta(spark, test_data, table_path)

    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 1
    assert filtered_df.filter(F.col("transaction_id") == "tx2").count() == 1


def test_filter_bin_length(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    test_data = [
        ("tx1", datetime(2023, 1, 1, 10, 0, 0), "12345", "US", False),     # BIN too short
        ("tx2", datetime(2023, 1, 1, 11, 0, 0), "123456789", "CA", False), # BIN too long
        ("tx3", datetime(2023, 1, 1, 12, 0, 0), "123456", "GB", False),    # Valid BIN (6)
        ("tx4", datetime(2023, 1, 1, 13, 0, 0), "1234567", "DE", False),   # Valid BIN (7)
        ("tx5", datetime(2023, 1, 1, 14, 0, 0), "12345678", "FR", False),  # Valid BIN (8)
    ]
    _write_test_data_to_delta(spark, test_data, table_path)

    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 3
    kept_ids = [row.transaction_id for row in filtered_df.select("transaction_id").collect()]
    assert "tx3" in kept_ids
    assert "tx4" in kept_ids
    assert "tx5" in kept_ids


def test_all_filters_applied_valid_rows_kept(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)
    
    test_data = [
        # Valid cases
        ("valid1", datetime(2023, 1, 1, 10, 0, 0), "123456", "US", False),
        ("valid2", datetime(2023, 1, 1, 11, 0, 0), "9876543", "GB", False),
        # Invalid cases
        ("invalid_fraud", datetime(2023, 1, 1, 12, 0, 0), "111222", "CA", True),
        ("invalid_null_country", datetime(2023, 1, 1, 13, 0, 0), "333444", None, False),
        ("invalid_bin_short", datetime(2023, 1, 1, 14, 0, 0), "555", "DE", False),
        ("invalid_bin_long", datetime(2023, 1, 1, 15, 0, 0), "666777888", "FR", False),
    ]
    _write_test_data_to_delta(spark, test_data, table_path)

    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 2
    valid_ids = [row.transaction_id for row in filtered_df.select("transaction_id").collect()]
    assert "valid1" in valid_ids
    assert "valid2" in valid_ids


def test_empty_input_table(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    _write_test_data_to_delta(spark, [], table_path) # Write empty data

    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 0
    assert filtered_df.schema == TXN_SCHEMA


def test_no_valid_data_in_table(spark_session, monkeypatch):
    spark, test_delta_root, table_path = spark_session
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    test_data = [
        ("invalid_fraud", datetime(2023, 1, 1, 12, 0, 0), "111222", "CA", True),
        ("invalid_null_country", datetime(2023, 1, 1, 13, 0, 0), "333444", None, False),
        ("invalid_bin_short", datetime(2023, 1, 1, 14, 0, 0), "555", "DE", False),
    ]
    _write_test_data_to_delta(spark, test_data, table_path)

    filtered_df = load_filtered(spark, "2023-01")
    assert filtered_df.count() == 0


def test_load_filtered_table_not_exists(spark_session, monkeypatch):
    """
    Tests that load_filtered returns an empty DataFrame with TXN_SCHEMA
    if the source Delta table does not exist.
    The load_filtered function is designed to catch AnalysisException and do this.
    """
    spark, test_delta_root, table_path = spark_session # table_path is .../TRANSACTIONS_TABLE_NAME
    
    # Ensure the specific table path for this test does NOT exist
    # The test_delta_root is created by fixture, but table_path within it might not exist
    # or might have been created by a previous test.
    # For this test, we need to ensure the table *path* itself is removed.
    # The fixture creates the *parent* of table_path.
    
    # The dummy table path is f"{test_delta_root}/{TRANSACTIONS_TABLE_NAME}"
    if os.path.exists(table_path):
        shutil.rmtree(table_path)
    
    # Monkeypatch DELTA_LAKE_PATH so load_filtered looks inside our test_delta_root
    monkeypatch.setattr('config.DELTA_LAKE_PATH', test_delta_root)

    # Call load_filtered, expecting it to handle the missing table
    filtered_df = load_filtered(spark, "2023-01")

    assert filtered_df is not None
    assert filtered_df.count() == 0
    assert filtered_df.schema == TXN_SCHEMA
    # Add check for logged warning (requires log capture setup, more advanced)
    # For now, structural check is primary.
```
