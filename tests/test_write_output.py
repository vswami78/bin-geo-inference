"""
Unit tests for the write_output.py module.
"""
import pytest
import os
import shutil
import tempfile
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType, IntegerType
from pyspark.sql.utils import AnalysisException


# Assuming 'write_output.py' and other project files are structured to be importable.
from write_output import write_delta
# Need to import config values to be monkeypatched or to know what write_delta will use.
import config 
# Using the schema from compare_vendor as the input to write_delta
try:
    from compare_vendor import FLAG_DISCREPANCIES_OUTPUT_SCHEMA
except ImportError:
    logging.warning("Could not import FLAG_DISCREPANCIES_OUTPUT_SCHEMA from compare_vendor. Using fallback for tests.")
    FLAG_DISCREPANCIES_OUTPUT_SCHEMA = StructType([
        StructField("BIN", StringType(), True), StructField("inferred_top_country", StringType(), True),
        StructField("vendor_country", StringType(), True), StructField("is_discrepant", BooleanType(), False),
        StructField("bin_total", LongType(), True)
    ])


@pytest.fixture(scope="function") # Function scope for temp dir isolation
def spark_session_for_writing(monkeypatch):
    """
    Pytest fixture to create a SparkSession for Delta writing tests.
    - Creates a unique temporary directory for DELTA_LAKE_PATH for each test.
    - Monkeypatches config.DELTA_LAKE_PATH and config.INFERRED_BIN_COUNTRY_MAP_TABLE_NAME
      to use this temporary path and a test table name.
    - Cleans up the temporary directory after the test.
    """
    temp_dir = tempfile.mkdtemp(prefix="delta_write_tests_")
    test_table_name = "test_output_table"

    # Monkeypatch the config values that write_delta will use
    monkeypatch.setattr(config, 'DELTA_LAKE_PATH', temp_dir)
    monkeypatch.setattr(config, 'INFERRED_BIN_COUNTRY_MAP_TABLE_NAME', test_table_name)
    
    # This is the actual path write_delta will construct and use
    constructed_table_path = os.path.join(temp_dir, test_table_name)

    spark = SparkSession.builder \
        .appName("WriteDeltaTests") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()

    yield spark, constructed_table_path # Provide Spark and the full expected table path

    spark.stop()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    logging.info(f"Cleaned up temporary Delta directory: {temp_dir}")


def test_basic_write_and_readback(spark_session_for_writing):
    spark, table_path = spark_session_for_writing
    
    data = [("B1", "US", "CA", True, 100L), ("B2", "DE", "DE", False, 200L)]
    df_to_write = spark.createDataFrame(data, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    month_to_write = "2023-01"

    write_delta(spark, df_to_write, month_to_write)

    # Verify by reading back
    df_read = spark.read.format("delta").load(table_path)
    assert df_read.count() == 2
    assert "snapshot_month" in df_read.columns
    assert df_read.filter(F.col("snapshot_month") == month_to_write).count() == 2
    
    # Check some data
    row1 = df_read.filter(F.col("BIN") == "B1").first()
    assert row1 is not None
    assert row1["inferred_top_country"] == "US"
    assert row1["snapshot_month"] == month_to_write


def test_partitioning_multiple_months(spark_session_for_writing):
    spark, table_path = spark_session_for_writing

    data_m1 = [("B1_M1", "US", "CA", True, 100L)]
    df_m1 = spark.createDataFrame(data_m1, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    month_m1 = "2023-01"
    write_delta(spark, df_m1, month_m1)

    data_m2 = [("B1_M2", "DE", "DE", False, 200L)]
    df_m2 = spark.createDataFrame(data_m2, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    month_m2 = "2023-02"
    write_delta(spark, df_m2, month_m2)

    # Verify full table
    df_full = spark.read.format("delta").load(table_path)
    assert df_full.count() == 2
    
    # Verify specific partition read for month1
    df_read_m1 = df_full.filter(F.col("snapshot_month") == month_m1)
    assert df_read_m1.count() == 1
    assert df_read_m1.first()["BIN"] == "B1_M1"

    # Verify specific partition read for month2
    df_read_m2 = df_full.filter(F.col("snapshot_month") == month_m2)
    assert df_read_m2.count() == 1
    assert df_read_m2.first()["BIN"] == "B1_M2"
    
    # Check actual directory structure (optional, but good for sanity)
    # Path should be table_path / snapshot_month=2023-01 / part-*.parquet
    assert os.path.exists(os.path.join(table_path, f"snapshot_month={month_m1}"))
    assert os.path.exists(os.path.join(table_path, f"snapshot_month={month_m2}"))


def test_dynamic_partition_overwrite(spark_session_for_writing):
    spark, table_path = spark_session_for_writing

    # Month 1 - initial write
    data_m1_v1 = [("B1", "US", "CA", True, 100L), ("B2", "FR", "FR", False, 50L)]
    df_m1_v1 = spark.createDataFrame(data_m1_v1, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_m1_v1, "2023-01")

    # Month 2 - initial write
    data_m2 = [("B3", "DE", "DE", False, 200L)]
    df_m2 = spark.createDataFrame(data_m2, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_m2, "2023-02")

    # Month 1 - overwrite with new data
    data_m1_v2 = [("B1", "US", "US", False, 120L), ("B4", "GB", None, False, 80L)] # B2 from v1 is gone
    df_m1_v2 = spark.createDataFrame(data_m1_v2, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_m1_v2, "2023-01")

    df_full = spark.read.format("delta").load(table_path)
    df_full.show(truncate=False)
    assert df_full.count() == 3 # 2 from new M1, 1 from M2

    # Check M1 data (should be v2)
    m1_rows = df_full.filter(F.col("snapshot_month") == "2023-01").collect()
    assert len(m1_rows) == 2
    m1_bins = [row.BIN for row in m1_rows]
    assert "B1" in m1_bins
    assert "B4" in m1_bins
    assert "B2" not in m1_bins # B2 was in v1 of M1, should be gone

    # Check M2 data (should be untouched)
    m2_rows = df_full.filter(F.col("snapshot_month") == "2023-02").collect()
    assert len(m2_rows) == 1
    assert m2_rows[0]["BIN"] == "B3"


def test_schema_overwrite(spark_session_for_writing):
    spark, table_path = spark_session_for_writing

    # Initial schema and data
    data_v1 = [("B1", "US", "CA", True, 100L)]
    df_v1 = spark.createDataFrame(data_v1, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_v1, "2023-01")

    # Evolved schema (add a new column)
    evolved_schema = FLAG_DISCREPANCIES_OUTPUT_SCHEMA.add(StructField("new_col", IntegerType(), True))
    data_v2 = [("B1", "US", "CA", True, 120L, 999)] # Data for evolved schema
    df_v2 = spark.createDataFrame(data_v2, schema=evolved_schema)
    write_delta(spark, df_v2, "2023-01") # Overwrite same partition with new schema

    df_read = spark.read.format("delta").load(table_path)
    df_read.printSchema()
    df_read.show()
    
    assert "new_col" in df_read.columns
    assert df_read.count() == 1
    row = df_read.first()
    assert row["new_col"] == 999
    assert row["bin_total"] == 120 # Check data also updated

    # Write to another partition with original schema to see if new_col is null there
    data_m2_orig_schema = [("B3", "DE", "DE", False, 200L)]
    df_m2_orig_schema = spark.createDataFrame(data_m2_orig_schema, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_m2_orig_schema, "2023-02")
    
    df_read_final = spark.read.format("delta").load(table_path)
    df_read_final.show()
    assert df_read_final.where("snapshot_month = '2023-02'").first()["new_col"] is None


def test_empty_dataframe_write(spark_session_for_writing):
    spark, table_path = spark_session_for_writing
    
    empty_df = spark.createDataFrame([], schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    month_to_write = "2023-01"

    write_delta(spark, empty_df, month_to_write)

    # Verify by reading back
    df_read = spark.read.format("delta").load(table_path)
    # The table exists, but this partition should be empty
    assert df_read.filter(F.col("snapshot_month") == month_to_write).count() == 0
    
    # Check if table is empty or if it has other partitions (it shouldn't here)
    assert df_read.count() == 0 

    # Write something to another partition to make sure the empty partition for 2023-01 is truly there
    data_m2 = [("B_M2", "DE", "DE", False, 200L)]
    df_m2 = spark.createDataFrame(data_m2, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    write_delta(spark, df_m2, "2023-02")
    
    df_read_after_m2 = spark.read.format("delta").load(table_path)
    df_read_after_m2.show()
    assert df_read_after_m2.count() == 1 # Only M2 data
    assert df_read_after_m2.filter(F.col("snapshot_month") == month_to_write).count() == 0
    assert df_read_after_m2.filter(F.col("snapshot_month") == "2023-02").count() == 1


def test_invalid_month_format(spark_session_for_writing):
    spark, _ = spark_session_for_writing # table_path not needed as it should fail before write
    data = [("B1", "US", "CA", True, 100L)]
    df_to_write = spark.createDataFrame(data, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    
    invalid_months = ["2023/01", "202301", "23-01", "2023-1", "2023-001"]
    for invalid_month in invalid_months:
        with pytest.raises(ValueError, match="Invalid month format"):
            write_delta(spark, df_to_write, invalid_month)

def test_output_path_and_table_name_usage(spark_session_for_writing):
    """Verify that the function attempts to write to the correct path derived from config."""
    spark, expected_table_path = spark_session_for_writing 
    # expected_table_path is already constructed using the monkeypatched config values
    # (temp_dir / test_table_name)
    
    data = [("B1", "US", "CA", True, 100L)]
    df_to_write = spark.createDataFrame(data, schema=FLAG_DISCREPANCIES_OUTPUT_SCHEMA)
    month = "2023-01"
    
    write_delta(spark, df_to_write, month)
    
    assert os.path.exists(expected_table_path), f"Delta table not found at expected path: {expected_table_path}"
    # Check for a common Delta log file
    assert os.path.exists(os.path.join(expected_table_path, "_delta_log")), "Delta log not found"

```
