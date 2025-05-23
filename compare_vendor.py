"""
Module for comparing inferred BIN-country mappings with vendor-provided mappings.
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType

# Assuming config.py is in the same directory or Python path
try:
    # VENDOR_MAPPING_TABLE_NAME might be used by a loader function, not directly here.
    from config import VENDOR_MAPPING_TABLE_NAME 
except ImportError:
    logging.warning("Could not import VENDOR_MAPPING_TABLE_NAME from config. Using default.")
    VENDOR_MAPPING_TABLE_NAME = "vendor_bin_country_map_default" 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected output schema for flag_discrepancies function
FLAG_DISCREPANCIES_OUTPUT_SCHEMA = StructType([
    StructField("BIN", StringType(), True),
    StructField("inferred_top_country", StringType(), True), # Non-null as it comes from inferred_df's top country
    StructField("vendor_country", StringType(), True), # Nullable if BIN not in vendor_df or vendor_country itself is null
    StructField("is_discrepant", BooleanType(), False), # False if vendor_country is null or matches inferred
    StructField("bin_total", LongType(), True) # From inferred_df, useful context, can be null if original BIN had no total
])


def flag_discrepancies(inferred_df: DataFrame, vendor_df: DataFrame) -> DataFrame:
    """
    Compares inferred BIN-country mappings with vendor-provided mappings to flag discrepancies.

    The top inferred country is determined by the first row per BIN in `inferred_df`
    after ordering by transaction count descending and country ascending (for ties).

    Args:
        inferred_df: DataFrame with inferred BIN-country mappings. Expected to have
                     columns like "BIN", "country", "txn_count", "bin_total".
                     Typically the output of coverage.filter_supported_bins.
        vendor_df: DataFrame with vendor BIN-country mappings. Expected to have
                   at least "BIN" and "country" (which will be aliased to "vendor_country").

    Returns:
        A DataFrame with columns "BIN", "inferred_top_country", "vendor_country",
        "is_discrepant", and "bin_total", ordered by BIN.
    """
    inferred_rows = inferred_df.count()
    vendor_rows = vendor_df.count()
    logging.info(f"flag_discrepancies: Input inferred_df has {inferred_rows} rows.")
    logging.info(f"flag_discrepancies: Input vendor_df has {vendor_rows} rows.")

    active_spark_session = SparkSession.getActiveSession()
    if not active_spark_session:
        logging.error("flag_discrepancies: No active SparkSession.")
        raise RuntimeError("SparkSession not found in flag_discrepancies.")

    if inferred_rows == 0:
        logging.info("flag_discrepancies: inferred_df is empty. Returning empty DataFrame with defined schema.")
        return active_spark_session.createDataFrame([], FLAG_DISCREPANCIES_OUTPUT_SCHEMA)

    # 1. Determine the top inferred country for each BIN from inferred_df.
    # The input inferred_df (from coverage module) should be ordered by BIN, 
    # then by preference (e.g., txn_count desc, country asc).
    # We use row_number() to be absolutely sure we pick the top one according to this order.
    # Required columns in inferred_df: BIN, country, txn_count, bin_total
    # (cumulative_pct, country_pct may also be present but not directly used here for ranking)
    
    # Check for necessary columns in inferred_df
    required_inferred_cols = {"BIN", "country", "txn_count", "bin_total"}
    if not required_inferred_cols.issubset(inferred_df.columns):
        missing_cols = required_inferred_cols - set(inferred_df.columns)
        raise ValueError(f"Input inferred_df is missing required columns: {missing_cols}")

    window_spec_top_country = Window.partitionBy("BIN") \
                                    .orderBy(F.desc("txn_count"), F.asc("country"))

    top_inferred_countries_df = inferred_df.withColumn("rn", F.row_number().over(window_spec_top_country)) \
                                           .filter(F.col("rn") == 1) \
                                           .select(
                                               F.col("BIN"),
                                               F.col("country").alias("inferred_top_country"),
                                               F.col("bin_total") 
                                           )
    
    unique_inferred_bins = top_inferred_countries_df.count()
    logging.info(f"flag_discrepancies: Extracted {unique_inferred_bins} unique top inferred BIN-country pairs.")

    # Check for necessary columns in vendor_df
    required_vendor_cols = {"BIN", "country"}
    if not required_vendor_cols.issubset(vendor_df.columns):
        missing_cols = required_vendor_cols - set(vendor_df.columns)
        raise ValueError(f"Input vendor_df is missing required columns: {missing_cols} (expected 'country' for vendor's mapping).")
        
    # Rename vendor's country column to avoid ambiguity before join
    vendor_df_renamed = vendor_df.select(F.col("BIN"), F.col("country").alias("vendor_country"))

    # 2. Join with vendor_df.
    comparison_df = top_inferred_countries_df.join(
        vendor_df_renamed,
        "BIN",
        "left"
    )
    
    # 3. Add 'is_discrepant' column.
    # True if inferred_top_country is different from vendor_country AND vendor_country is not null.
    # Otherwise False (covers cases where they match, or where vendor_country is null).
    final_df = comparison_df.withColumn(
        "is_discrepant",
        F.when(
            (F.col("inferred_top_country") != F.col("vendor_country")) & F.col("vendor_country").isNotNull(),
            True
        ).otherwise(False)
    )

    # Select and order final columns as per FLAG_DISCREPANCIES_OUTPUT_SCHEMA
    final_df = final_df.select(
        FLAG_DISCREPANCIES_OUTPUT_SCHEMA.fieldNames() # Select in schema order
    ).orderBy("BIN")


    output_rows = final_df.count()
    discrepant_count = final_df.filter(F.col("is_discrepant") == True).count()
    logging.info(f"flag_discrepancies: Output DataFrame has {output_rows} rows.")
    logging.info(f"flag_discrepancies: Found {discrepant_count} discrepant BINs.")
    
    return final_df

if __name__ == '__main__':
    local_spark = SparkSession.builder \
        .appName("CompareVendorLocalTest") \
        .master("local[*]") \
        .getOrCreate()

    # Sample data for inferred_df (simulating output of coverage module)
    try:
        from coverage import SELECT_TOP_COUNTRIES_SCHEMA as INFERRED_SCHEMA
    except ImportError:
        logging.warning("Could not import SELECT_TOP_COUNTRIES_SCHEMA from coverage. Using fallback for __main__.")
        INFERRED_SCHEMA = StructType([
            StructField("BIN", StringType(), True), StructField("country", StringType(), True),
            StructField("txn_count", LongType(), False), StructField("bin_total", LongType(), False),
            StructField("country_pct", DoubleType(), False), StructField("cumulative_pct", DoubleType(), False)
        ])

    sample_inferred_data = [
        ("BIN1", "US", 60, 100, 0.6, 0.6), ("BIN1", "CA", 30, 100, 0.3, 0.9), 
        ("BIN2", "DE", 90, 100, 0.9, 0.9), ("BIN2", "FR", 10, 100, 0.1, 1.0),
        ("BIN3", "JP", 100, 100, 1.0, 1.0),
        ("BIN4", "AU", 120, 120, 1.0, 1.0),
        ("BIN5", "GB", 50, 50, 1.0, 1.0),
        # BIN7: multiple rows, ensure top one (US by txn_count) is picked
        ("BIN7", "US", 70, 150, 0.466, 0.466),
        ("BIN7", "CA", 50, 150, 0.333, 0.799),
        ("BIN7", "MX", 30, 150, 0.200, 0.999),
    ]
    inferred_test_df = local_spark.createDataFrame(sample_inferred_data, schema=INFERRED_SCHEMA)

    vendor_schema = StructType([
        StructField("BIN", StringType(), True),
        StructField("country", StringType(), True) 
    ])
    sample_vendor_data = [
        ("BIN1", "CA"), 
        ("BIN2", "DE"), 
        ("BIN4", "AU"), 
        ("BIN5", "gb"), # Test case sensitivity: "GB" vs "gb" will be a discrepancy
        ("BIN6", "US"), 
        ("BIN7", "US"), # BIN7 matches inferred top
    ]
    vendor_test_df = local_spark.createDataFrame(sample_vendor_data, schema=vendor_schema)

    logging.info("CompareVendorLocalTest (__main__): Input inferred_df (first few rows):")
    inferred_test_df.show(5, truncate=False)
    logging.info("CompareVendorLocalTest (__main__): Input vendor_df:")
    vendor_test_df.show(truncate=False)

    comparison_result_df = flag_discrepancies(inferred_test_df, vendor_test_df)

    logging.info("CompareVendorLocalTest (__main__): Comparison Result:")
    comparison_result_df.show(truncate=False)
    
    # Expected for BIN1: inferred US, vendor CA -> True, bin_total 100
    # Expected for BIN5: inferred GB, vendor gb -> True, bin_total 50
    # Expected for BIN7: inferred US, vendor US -> False, bin_total 150

    bin1_res = comparison_result_df.filter("BIN = 'BIN1'").first()
    assert bin1_res["is_discrepant"] is True and bin1_res["vendor_country"] == "CA"
    
    bin5_res = comparison_result_df.filter("BIN = 'BIN5'").first()
    assert bin5_res["is_discrepant"] is True and bin5_res["vendor_country"] == "gb"

    bin7_res = comparison_result_df.filter("BIN = 'BIN7'").first()
    assert bin7_res["is_discrepant"] is False and bin7_res["inferred_top_country"] == "US" and bin7_res["vendor_country"] == "US"
    
    logging.info("CompareVendorLocalTest (__main__): Ad-hoc assertions passed.")
    local_spark.stop()
```
