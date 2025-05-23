"""
Configuration file for the BIN geolocation project.
"""

from pyspark.sql.types import StructType, StructField, StringType, BooleanType, TimestampType

# Global constants
MIN_TXNS: int = 100  # Minimum transactions required for a BIN to be considered for geolocation inference
COVERAGE_PCT: float = 0.95  # Required coverage percentage for a country to be confidently mapped to a BIN

TRANSACTIONS_TABLE_NAME: str = "transactions"  # Name of the input transactions Delta table
INFERRED_BIN_COUNTRY_MAP_TABLE_NAME: str = "bin_country_map_inferred"  # Name of the output Delta table for inferred BIN-country mappings
VENDOR_MAPPING_TABLE_NAME: str = "vendor_bin_country_map"  # Placeholder for a potential vendor-supplied BIN-country mapping table
DELTA_LAKE_PATH: str = "/mnt/delta/bin_geolocation"  # Base path for Delta Lake tables related to this project

# Spark DataFrame schema for input transaction data
# Based on spec.md: BIN, Billing Address (country), Fraud label (true/false)
TXN_SCHEMA: StructType = StructType([
    StructField("transaction_id", StringType(), True),      # Unique identifier for the transaction (assumed, good practice)
    StructField("timestamp", TimestampType(), True),        # Timestamp of the transaction (assumed, good practice)
    StructField("BIN", StringType(), False),                # Bank Identification Number from the transaction log
    StructField("billing_address_country", StringType(), True),  # Country derived from the billing address
    StructField("fraud_label", BooleanType(), True)         # Label indicating if the transaction was marked as fraudulent
])

# Example of how to use these constants in another file:
#
# from config import MIN_TXNS, TRANSACTIONS_TABLE_NAME, TXN_SCHEMA, DELTA_LAKE_PATH
#
# print(f"Minimum transactions for inference: {MIN_TXNS}")
# print(f"Input transactions table: {TRANSACTIONS_TABLE_NAME}")
# print(f"Output inferred BIN-country map table: {INFERRED_BIN_COUNTRY_MAP_TABLE_NAME}")
# print(f"Transaction schema definition: {TXN_SCHEMA}")
# print(f"Delta Lake base path: {DELTA_LAKE_PATH}")
