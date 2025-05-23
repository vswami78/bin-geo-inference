"""
Unit tests for the compare_vendor.py module.
"""
import pytest

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, BooleanType

# Assuming 'compare_vendor.py' and 'coverage.py' are importable.
from compare_vendor import flag_discrepancies, FLAG_DISCREPANCIES_OUTPUT_SCHEMA
from coverage import SELECT_TOP_COUNTRIES_SCHEMA # For creating inferred_df test data

# Define a schema for vendor_df for test purposes
VENDOR_SCHEMA = StructType([
    StructField("BIN", StringType(), True),
    StructField("country", StringType(), True) # This will be aliased to vendor_country by the function
])

@pytest.fixture(scope="session")
def spark_session():
    """Pytest fixture for creating a SparkSession (session-scoped)."""
    spark = SparkSession.builder \
        .appName("CompareVendorTests") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "1") 
        .getOrCreate()
    yield spark
    spark.stop()

# Test Data - common structure for inferred_df (using SELECT_TOP_COUNTRIES_SCHEMA)
# BIN, country, txn_count, bin_total, country_pct, cumulative_pct

def test_basic_discrepancy(spark_session):
    """inferred_top_country differs from vendor_country."""
    inferred_data = [("BIN1", "US", 100, 100, 1.0, 1.0)]
    vendor_data = [("BIN1", "CA")] # Vendor says CA
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 1
    row = result_df.first()
    assert row["BIN"] == "BIN1"
    assert row["inferred_top_country"] == "US"
    assert row["vendor_country"] == "CA"
    assert row["is_discrepant"] is True
    assert row["bin_total"] == 100

def test_no_discrepancy(spark_session):
    """inferred_top_country matches vendor_country."""
    inferred_data = [("BIN1", "US", 100, 100, 1.0, 1.0)]
    vendor_data = [("BIN1", "US")]
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()

    assert result_df.count() == 1
    row = result_df.first()
    assert row["is_discrepant"] is False
    assert row["vendor_country"] == "US"

def test_top_inferred_country_selection(spark_session):
    """Ensures only the top inferred country is used for comparison when multiple exist."""
    inferred_data = [
        ("BIN_MULTI", "US", 70, 100, 0.7, 0.7), # Top country by txn_count
        ("BIN_MULTI", "CA", 30, 100, 0.3, 1.0), # Lower txn_count
    ]
    vendor_data = [("BIN_MULTI", "US")] # Vendor matches the top inferred
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 1 # One row per BIN in output
    row = result_df.first()
    assert row["BIN"] == "BIN_MULTI"
    assert row["inferred_top_country"] == "US" # Check that US was chosen
    assert row["vendor_country"] == "US"
    assert row["is_discrepant"] is False

def test_bin_in_inferred_not_in_vendor(spark_session):
    """BIN in inferred_df, not in vendor_df. is_discrepant should be False."""
    inferred_data = [("BIN_ONLY_INF", "DE", 50, 50, 1.0, 1.0)]
    vendor_data = [] # Empty vendor data
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 1
    row = result_df.first()
    assert row["BIN"] == "BIN_ONLY_INF"
    assert row["inferred_top_country"] == "DE"
    assert row["vendor_country"] is None
    assert row["is_discrepant"] is False # As per logic: (cond1 & cond2_isNotNull)

def test_bin_in_vendor_not_in_inferred(spark_session):
    """BIN in vendor_df, not in inferred_df. Should not appear in output."""
    inferred_data = [] # Empty inferred data
    vendor_data = [("BIN_ONLY_VENDOR", "FR")]
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 0 # Because of left join from (empty) inferred

def test_empty_inferred_df(spark_session):
    """Test with empty inferred_df."""
    inferred_data = []
    vendor_data = [("BIN1", "US")]
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 0
    assert result_df.schema == FLAG_DISCREPANCIES_OUTPUT_SCHEMA

def test_empty_vendor_df(spark_session):
    """Test with empty vendor_df. All vendor_country should be null, no discrepancies."""
    inferred_data = [
        ("BIN1", "US", 100, 100, 1.0, 1.0),
        ("BIN2", "CA", 50, 50, 1.0, 1.0),
    ]
    vendor_data = []
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    assert result_df.count() == 2
    for row in result_df.collect():
        assert row["vendor_country"] is None
        assert row["is_discrepant"] is False

def test_output_schema_validation(spark_session):
    """Verify the output schema."""
    inferred_data = [("BIN1", "US", 100, 100, 1.0, 1.0)]
    vendor_data = [("BIN1", "CA")]
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    
    assert result_df.schema == FLAG_DISCREPANCIES_OUTPUT_SCHEMA

def test_case_insensitivity_not_handled_by_default(spark_session):
    """Test if 'US' vs 'us' is treated as a discrepancy."""
    inferred_data = [("BIN_CASE", "US", 100, 100, 1.0, 1.0)]
    vendor_data = [("BIN_CASE", "us")] # Vendor has lowercase 'us'
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    row = result_df.first()
    assert row["is_discrepant"] is True # Default string comparison is case-sensitive

def test_vendor_country_is_null_in_vendor_file(spark_session):
    """Test when a BIN exists in vendor_df but its 'country' field is null."""
    inferred_data = [("BIN_NULL_VENDOR", "US", 100, 100, 1.0, 1.0)]
    # Vendor data where country is explicitly None (null)
    vendor_data_with_null = [Row(BIN="BIN_NULL_VENDOR", country=None)]
    
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data_with_null, schema=VENDOR_SCHEMA)
    
    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()
    
    row = result_df.first()
    assert row["BIN"] == "BIN_NULL_VENDOR"
    assert row["inferred_top_country"] == "US"
    assert row["vendor_country"] is None
    assert row["is_discrepant"] is False # Because vendor_country.isNotNull() is false

def test_multiple_matches_and_discrepancies(spark_session):
    inferred_data = [
        ("BIN1", "US", 100, 100, 1.0, 1.0), # Match
        ("BIN2", "CA", 80, 100, 0.8, 0.8),  # Discrepancy
        ("BIN2", "US", 20, 100, 0.2, 1.0),
        ("BIN3", "GB", 100, 100, 1.0, 1.0), # Not in vendor
        ("BIN4", "DE", 70, 70, 1.0, 1.0),   # Match
    ]
    vendor_data = [
        ("BIN1", "US"),
        ("BIN2", "MX"), # Vendor says MX for BIN2
        ("BIN4", "DE"),
        ("BIN5", "FR"), # Only in vendor
    ]
    inferred_df = spark_session.createDataFrame(inferred_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    vendor_df = spark_session.createDataFrame(vendor_data, schema=VENDOR_SCHEMA)

    result_df = flag_discrepancies(inferred_df, vendor_df)
    result_df.show()

    results_map = {row.BIN: row for row in result_df.collect()}
    
    assert len(results_map) == 4 # BIN1, BIN2, BIN3, BIN4

    assert results_map["BIN1"].is_discrepant is False
    assert results_map["BIN1"].inferred_top_country == "US"
    assert results_map["BIN1"].vendor_country == "US"

    assert results_map["BIN2"].is_discrepant is True
    assert results_map["BIN2"].inferred_top_country == "CA" # Top from inferred_data for BIN2
    assert results_map["BIN2"].vendor_country == "MX"

    assert results_map["BIN3"].is_discrepant is False
    assert results_map["BIN3"].inferred_top_country == "GB"
    assert results_map["BIN3"].vendor_country is None

    assert results_map["BIN4"].is_discrepant is False
    assert results_map["BIN4"].inferred_top_country == "DE"
    assert results_map["BIN4"].vendor_country == "DE"

```
