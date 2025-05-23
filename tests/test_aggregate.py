"""
Unit tests for the aggregate.py module.
"""
import pytest
from datetime import datetime

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F

from aggregate import country_counts, COUNTRY_COUNTS_SCHEMA
from config import TXN_SCHEMA # Used for creating realistic input data

@pytest.fixture(scope="session")
def spark_session():
    """Pytest fixture to create a SparkSession for testing (session-scoped)."""
    spark = SparkSession.builder \
        .appName("AggregateTests") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark
    spark.stop()

# Sample data for tests - using all fields from TXN_SCHEMA for consistency
# transaction_id, timestamp, BIN, billing_address_country, fraud_label

def test_basic_aggregation_and_total(spark_session):
    """Test basic txn_count and bin_total calculation."""
    test_data = [
        ("tx1", datetime(2023,1,1,10,0,0), "123456", "US", False),
        ("tx2", datetime(2023,1,1,11,0,0), "123456", "US", False),
        ("tx3", datetime(2023,1,1,12,0,0), "123456", "CA", False),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    
    result_df = country_counts(input_df)
    result_df.show()

    assert result_df.count() == 2
    
    us_row = result_df.filter((F.col("BIN") == "123456") & (F.col("country") == "US")).first()
    assert us_row is not None
    assert us_row["txn_count"] == 2
    assert us_row["bin_total"] == 3
    
    ca_row = result_df.filter((F.col("BIN") == "123456") & (F.col("country") == "CA")).first()
    assert ca_row is not None
    assert ca_row["txn_count"] == 1
    assert ca_row["bin_total"] == 3

def test_single_bin_multiple_countries(spark_session):
    """Test one BIN with transactions in multiple countries."""
    test_data = [
        ("t1", datetime(2023,1,1,10,0,0), "BIN_A", "US", False),
        ("t2", datetime(2023,1,1,11,0,0), "BIN_A", "US", False),
        ("t3", datetime(2023,1,1,12,0,0), "BIN_A", "CA", False),
        ("t4", datetime(2023,1,1,13,0,0), "BIN_A", "GB", False),
        ("t5", datetime(2023,1,1,14,0,0), "BIN_A", "GB", False),
        ("t6", datetime(2023,1,1,15,0,0), "BIN_A", "GB", False),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    result_df = country_counts(input_df)
    result_df.show()

    assert result_df.count() == 3 # BIN_A in US, CA, GB
    
    # Check total for BIN_A
    # All rows for BIN_A should have the same bin_total
    bin_a_rows = result_df.filter(F.col("BIN") == "BIN_A").collect()
    assert len(bin_a_rows) == 3
    for row in bin_a_rows:
        assert row["bin_total"] == 6 # 2 (US) + 1 (CA) + 3 (GB)

    # Check specific country counts for BIN_A
    us_count = result_df.filter(F.col("country") == "US").first()["txn_count"]
    ca_count = result_df.filter(F.col("country") == "CA").first()["txn_count"]
    gb_count = result_df.filter(F.col("country") == "GB").first()["txn_count"]
    assert us_count == 2
    assert ca_count == 1
    assert gb_count == 3

def test_multiple_bins_varied_data(spark_session):
    """Test with multiple BINs and varied transaction distributions."""
    test_data = [
        # BIN_X
        ("x1", datetime(2023,1,1,10,0,0), "BIN_X", "US", False),
        ("x2", datetime(2023,1,1,11,0,0), "BIN_X", "US", False),
        # BIN_Y
        ("y1", datetime(2023,1,1,12,0,0), "BIN_Y", "CA", False),
        # BIN_Z
        ("z1", datetime(2023,1,1,13,0,0), "BIN_Z", "GB", False),
        ("z2", datetime(2023,1,1,14,0,0), "BIN_Z", "GB", False),
        ("z3", datetime(2023,1,1,15,0,0), "BIN_Z", "FR", False),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    result_df = country_counts(input_df)
    result_df.show()
    
    # Expected number of rows: BIN_X (US), BIN_Y (CA), BIN_Z (GB), BIN_Z (FR) -> 4 rows
    assert result_df.count() == 4
    
    # Collect data for easier assertion
    results = { (r["BIN"], r["country"]): (r["txn_count"], r["bin_total"]) 
                for r in result_df.collect() }

    assert results[("BIN_X", "US")] == (2, 2)
    assert results[("BIN_Y", "CA")] == (1, 1)
    assert results[("BIN_Z", "GB")] == (2, 3)
    assert results[("BIN_Z", "FR")] == (1, 3)
    
    # Check column names
    expected_cols = ["BIN", "country", "txn_count", "bin_total"]
    assert all(col in result_df.columns for col in expected_cols)
    assert len(result_df.columns) == len(expected_cols)


def test_empty_input(spark_session):
    """Test with an empty input DataFrame."""
    empty_df_with_schema = spark_session.createDataFrame([], schema=TXN_SCHEMA)
    result_df = country_counts(empty_df_with_schema)
    
    assert result_df.count() == 0
    # Check if the schema matches COUNTRY_COUNTS_SCHEMA defined in aggregate.py
    assert result_df.schema == COUNTRY_COUNTS_SCHEMA

def test_bin_with_single_transaction(spark_session):
    """Test a BIN that has only one transaction."""
    test_data = [
        ("single_tx", datetime(2023,1,1,10,0,0), "SINGLE_BIN", "DE", False),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    result_df = country_counts(input_df)
    result_df.show()

    assert result_df.count() == 1
    row = result_df.first()
    assert row is not None
    assert row["BIN"] == "SINGLE_BIN"
    assert row["country"] == "DE"
    assert row["txn_count"] == 1
    assert row["bin_total"] == 1

def test_all_transactions_one_country_for_bin(spark_session):
    """Test a BIN where all its transactions are in a single country."""
    test_data = [
        ("b1", datetime(2023,1,1,10,0,0), "ONE_COUNTRY_BIN", "JP", False),
        ("b2", datetime(2023,1,1,11,0,0), "ONE_COUNTRY_BIN", "JP", False),
        ("b3", datetime(2023,1,1,12,0,0), "ONE_COUNTRY_BIN", "JP", False),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    result_df = country_counts(input_df)
    result_df.show()

    assert result_df.count() == 1
    row = result_df.first()
    assert row is not None
    assert row["BIN"] == "ONE_COUNTRY_BIN"
    assert row["country"] == "JP"
    assert row["txn_count"] == 3
    assert row["bin_total"] == 3

def test_schema_and_column_names(spark_session):
    """Test the output schema and column names explicitly."""
    test_data = [("tx1", datetime(2023,1,1,10,0,0), "SCHEMA_TEST_BIN", "US", False)]
    input_df = spark_session.createDataFrame(test_data, schema=TXN_SCHEMA)
    result_df = country_counts(input_df)

    # Check against COUNTRY_COUNTS_SCHEMA from aggregate.py
    assert result_df.schema == COUNTRY_COUNTS_SCHEMA
    
    # Double check column names just in case schema matches but names are subtly different (e.g. case)
    # COUNTRY_COUNTS_SCHEMA.fieldNames() gives the correct names
    expected_column_names = COUNTRY_COUNTS_SCHEMA.fieldNames()
    assert result_df.columns == expected_column_names
```
