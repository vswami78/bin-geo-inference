"""
Unit tests for the coverage.py module.
"""
import pytest
from pytest import approx

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F

# Assuming 'coverage.py' and 'aggregate.py' are structured to be importable.
# If they are in the root and tests/ is a subdir, ensure root is in PYTHONPATH.
from coverage import select_top_countries, filter_supported_bins, SELECT_TOP_COUNTRIES_SCHEMA
from aggregate import COUNTRY_COUNTS_SCHEMA # For input data schema for select_top_countries
from config import COVERAGE_PCT, MIN_TXNS # To know the value being used in tests

@pytest.fixture(scope="session")
def spark_session():
    """Pytest fixture for creating a SparkSession (session-scoped)."""
    spark = SparkSession.builder \
        .appName("CoverageTests") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "2") # Small number for local testing
        .getOrCreate()
    yield spark
    spark.stop()

# Test Data - structured using COUNTRY_COUNTS_SCHEMA: (BIN, country, txn_count, bin_total)
# for select_top_countries tests.
# For filter_supported_bins tests, input data uses SELECT_TOP_COUNTRIES_SCHEMA.

def test_basic_coverage_selection(spark_session):
    """Test that countries are selected until cumulative_pct meets COVERAGE_PCT."""
    test_data = [
        ("BIN1", "US", 60, 100), ("BIN1", "CA", 30, 100), ("BIN1", "MX", 5, 100), ("BIN1", "GB", 5, 100),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    
    assert result_df.count() == 3
    kept_countries = [row.country for row in result_df.filter(F.col("BIN") == "BIN1").collect()]
    assert "US" in kept_countries and "CA" in kept_countries and "MX" in kept_countries
    assert "GB" not in kept_countries

def test_ordering_and_tie_breaking(spark_session):
    """Test correct selection with ties in txn_count, relying on country name tie-breaking."""
    test_data = [
        ("BIN_TIE", "C_Gamma", 40, 100), ("BIN_TIE", "C_Alpha", 30, 100), ("BIN_TIE", "C_Beta",  30, 100),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    
    assert result_df.count() == 3 
    results = result_df.filter(F.col("BIN") == "BIN_TIE").orderBy("cumulative_pct").collect()
    assert results[0].country == "C_Gamma" and results[1].country == "C_Alpha" and results[2].country == "C_Beta"
    assert results[0].cumulative_pct == approx(0.4) and results[1].cumulative_pct == approx(0.7) and results[2].cumulative_pct == approx(1.0)

def test_single_country_meets_coverage(spark_session):
    test_data = [("BIN_SINGLE", "US", 98, 100), ("BIN_SINGLE", "CA", 2, 100)]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    assert result_df.count() == 1 and result_df.first().country == "US"

def test_all_countries_needed_for_coverage(spark_session):
    test_data = [("BIN_LOW", "US", 10, 100), ("BIN_LOW", "CA", 10, 100), ("BIN_LOW", "GB", 10, 100)]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    assert result_df.count() == 3

def test_multiple_bins_different_distributions(spark_session):
    test_data = [
        ("BIN_A", "US", 70, 100), ("BIN_A", "CA", 30, 100),
        ("BIN_B", "DE", 96, 100), ("BIN_B", "FR", 4, 100),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    assert result_df.filter(F.col("BIN") == "BIN_A").count() == 2
    assert result_df.filter(F.col("BIN") == "BIN_B").count() == 1

def test_empty_input_df_select_top_countries(spark_session):
    empty_input_df = spark_session.createDataFrame([], COUNTRY_COUNTS_SCHEMA)
    result_df = select_top_countries(empty_input_df)
    assert result_df.count() == 0 and result_df.schema == SELECT_TOP_COUNTRIES_SCHEMA

def test_output_schema_select_top_countries(spark_session):
    test_data = [("BIN_SCHEMA", "US", 10, 20)]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    assert result_df.schema == SELECT_TOP_COUNTRIES_SCHEMA

def test_country_pct_and_cumulative_pct_values(spark_session):
    test_data = [("BIN_CALC", "US", 75, 100), ("BIN_CALC", "CA", 20, 100), ("BIN_CALC", "MX", 5, 100)]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    us_row = result_df.filter(F.col("country") == "US").first()
    ca_row = result_df.filter(F.col("country") == "CA").first()
    mx_row = result_df.filter(F.col("country") == "MX").first()
    assert us_row["country_pct"] == approx(0.75) and us_row["cumulative_pct"] == approx(0.75)
    assert ca_row["country_pct"] == approx(0.20) and ca_row["cumulative_pct"] == approx(0.95)
    assert mx_row["country_pct"] == approx(0.05) and mx_row["cumulative_pct"] == approx(1.0)

def test_coverage_pct_boundary_exact_match(spark_session):
    test_data = [("BIN_EXACT", "A", 90, 100), ("BIN_EXACT", "B", 5, 100), ("BIN_EXACT", "C", 5, 100)]
    input_df = spark_session.createDataFrame(test_data, schema=COUNTRY_COUNTS_SCHEMA)
    sorted_input_df = input_df.orderBy("BIN", F.desc("txn_count"), F.asc("country"))
    result_df = select_top_countries(sorted_input_df)
    assert result_df.count() == 2
    kept_countries = [row.country for row in result_df.collect()]
    assert "A" in kept_countries and "B" in kept_countries and "C" not in kept_countries
    assert result_df.filter(F.col("country") == "B").first()["cumulative_pct"] == approx(0.95)

# --- Tests for filter_supported_bins ---
# Input data for these tests should conform to SELECT_TOP_COUNTRIES_SCHEMA

def test_filter_supported_bins_basic(spark_session):
    """Test that BINs with bin_total < MIN_TXNS are removed."""
    # Assume MIN_TXNS is, for example, 100 (from config)
    # If MIN_TXNS is 10 (default in coverage.py if config fails), adjust test data
    # For this test, let's use a value clearly below a typical MIN_TXNS like 100 or default 10
    below_min_txns = MIN_TXNS - 1 if MIN_TXNS > 0 else 0 # e.g. 9 or 99
    
    test_data = [
        # BIN, country, txn_count, bin_total, country_pct, cumulative_pct
        ("BIN_SUPPORTED", "US", MIN_TXNS, MIN_TXNS, 1.0, 1.0),
        ("BIN_UNSUPPORTED", "CA", below_min_txns, below_min_txns, 1.0, 1.0),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    
    assert result_df.count() == 1
    assert result_df.filter(F.col("BIN") == "BIN_SUPPORTED").count() == 1
    assert result_df.filter(F.col("BIN") == "BIN_UNSUPPORTED").count() == 0

def test_filter_supported_bins_boundary(spark_session):
    """Test that BINs with bin_total == MIN_TXNS are kept."""
    test_data = [
        ("BIN_AT_MIN", "US", MIN_TXNS, MIN_TXNS, 1.0, 1.0),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    assert result_df.count() == 1
    assert result_df.first().BIN == "BIN_AT_MIN"

def test_filter_supported_bins_mixed(spark_session):
    """Test with multiple BINs, some above, at, and below the MIN_TXNS threshold."""
    below_min = MIN_TXNS - 1 if MIN_TXNS > 0 else 0
    above_min = MIN_TXNS + 1

    test_data = [
        ("BIN_ABOVE", "US", above_min, above_min, 1.0, 1.0),
        ("BIN_AT", "CA", MIN_TXNS, MIN_TXNS, 1.0, 1.0),
        ("BIN_BELOW", "GB", below_min, below_min, 1.0, 1.0),
        # Test a BIN with multiple country rows, but total is below threshold
        ("BIN_BELOW_MULTI", "DE", below_min -1 if below_min >0 else 0, below_min, (below_min-1)/below_min if below_min >0 else 0, (below_min-1)/below_min if below_min >0 else 0),
        ("BIN_BELOW_MULTI", "FR", 1, below_min, 1/below_min if below_min >0 else 0, 1.0 if below_min >0 else 0),
    ]
    # Filter out potentially invalid data for BIN_BELOW_MULTI if below_min is too small
    valid_test_data = [r for r in test_data if r[3] > 0 or (r[3]==0 and r[2]==0) ]


    input_df = spark_session.createDataFrame(valid_test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    
    kept_bins = [row.BIN for row in result_df.select("BIN").distinct().collect()]
    assert "BIN_ABOVE" in kept_bins
    assert "BIN_AT" in kept_bins
    assert "BIN_BELOW" not in kept_bins
    assert "BIN_BELOW_MULTI" not in kept_bins
    assert len(kept_bins) == 2

def test_filter_supported_bins_empty_input(spark_session):
    """Test filter_supported_bins with an empty input DataFrame."""
    empty_df = spark_session.createDataFrame([], SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(empty_df)
    assert result_df.count() == 0
    assert result_df.schema == SELECT_TOP_COUNTRIES_SCHEMA

def test_filter_supported_bins_all_above(spark_session):
    """Test when all BINs are above the MIN_TXNS threshold."""
    above_min1 = MIN_TXNS + 10
    above_min2 = MIN_TXNS + 20
    test_data = [
        ("BIN_HIGH1", "US", above_min1, above_min1, 1.0, 1.0),
        ("BIN_HIGH1", "CA", 0, above_min1, 0.0, 1.0), # ensure another row for same BIN
        ("BIN_HIGH2", "DE", above_min2, above_min2, 1.0, 1.0),
    ]
    input_df = spark_session.createDataFrame(test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    assert result_df.count() == 3 # All 3 rows should be kept
    assert input_df.select("BIN").distinct().count() == result_df.select("BIN").distinct().count()


def test_filter_supported_bins_all_below(spark_session):
    """Test when all BINs are below the MIN_TXNS threshold."""
    # Ensure bin_total is always positive for these tests to be meaningful if MIN_TXNS can be 0 or 1
    below_min1 = MIN_TXNS - 1 if MIN_TXNS > 1 else 1 
    below_min2 = MIN_TXNS - 2 if MIN_TXNS > 2 else 1
    
    # Make sure bin_total is at least 1 if MIN_TXNS is 1 or 2.
    # If MIN_TXNS = 100, then below_min1=99, below_min2=98
    # If MIN_TXNS = 1, then below_min1=1 (but filter is >=), so it means it should be 0.
    # Let's define "below" as strictly less than MIN_TXNS.
    # So if MIN_TXNS = 1, any bin_total of 0 is below.
    # If MIN_TXNS = 0, nothing is below.
    
    if MIN_TXNS == 0: # Edge case, all bins should be kept
        test_data = [("BIN_ANY", "US", 0, 0, 0.0, 0.0)] # Assuming 0/0 is 0 for pct
        input_df = spark_session.createDataFrame(test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
        result_df = filter_supported_bins(input_df)
        assert result_df.count() == 1
        return

    # General case where MIN_TXNS >= 1
    val1 = max(0, MIN_TXNS - 1)
    val2 = max(0, MIN_TXNS - 2)

    test_data = [
        ("BIN_LOW1", "US", val1, val1, 1.0 if val1 > 0 else 0.0, 1.0 if val1 > 0 else 0.0),
        ("BIN_LOW2", "DE", val2, val2, 1.0 if val2 > 0 else 0.0, 1.0 if val2 > 0 else 0.0),
    ]
    # If MIN_TXNS makes val1 or val2 zero, ensure txn_count is also zero
    cleaned_test_data = []
    for r in test_data:
        if r[3] == 0: # if bin_total is 0
            cleaned_test_data.append((r[0], r[1], 0, 0, 0.0, 0.0))
        else:
            cleaned_test_data.append(r)
            
    input_df = spark_session.createDataFrame(cleaned_test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    assert result_df.count() == 0

def test_filter_supported_bins_preserves_schema(spark_session):
    """Test that the schema is preserved by filter_supported_bins."""
    test_data = [("BIN_SCH", "US", MIN_TXNS, MIN_TXNS, 1.0, 1.0)]
    input_df = spark_session.createDataFrame(test_data, schema=SELECT_TOP_COUNTRIES_SCHEMA)
    result_df = filter_supported_bins(input_df)
    assert result_df.schema == SELECT_TOP_COUNTRIES_SCHEMA
```
