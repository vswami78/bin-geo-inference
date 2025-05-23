# spec.md

### `spec.md`: BIN Geolocation Inference Pipeline

#### **Overview**

We aim to infer the geographic origin (country) of credit card BINs using billing address data from non-fraudulent transactions. This will allow us to build a home-grown BIN-to-country mapping based on empirical usage patterns, compare against vendor data, and identify problematic BINs where there is significant disagreement.

---

#### **Requirements**

* Process only **non-fraudulent** transactions with **non-null billing addresses**
* Use **billing address country** for geo-inference
* For each BIN:

  * Capture **top N countries** that cumulatively explain ≥ 95% of usage
  * Include only BINs with ≥ *X* distinct transactions (threshold TBD)
* Output a **Databricks delta table** with BIN-to-country mappings
* Snapshot this output **monthly**, keeping version history
* Include comparison to current vendor-provided country mapping
* Flag BINs with significant discrepancies
* Incorporate **data quality checks**:

  * Valid country codes
  * BIN length (6–8 digits)
  * No duplicate BIN-country records
* MVP will **not** trigger automatic second opinions but leaves room for future vendor integration
* Must support full **monthly reprocessing** end-to-end without manual steps

---

#### **Architecture**

* **Source**: Internal transaction log with fields:

  * BIN
  * Billing Address (country)
  * Fraud label (true/false)
* **Pipeline Stages**:

  1. Filter and clean data
  2. Aggregate billing countries per BIN
  3. Compute top N countries covering ≥ 95% of usage
  4. Compare against 3rd-party vendor mapping
  5. Flag discrepancies
  6. Save results to versioned delta table

---

#### **Data Handling**

* Input:

  * Databricks table: `transactions`
* Output:

  * Databricks delta table: `bin_country_map_inferred`
* Temporary:

  * Intermediate aggregation and discrepancy logs

---

#### **Error Cases**

* Missing or malformed BINs
* Country field null or invalid (e.g. malformed ISO codes)
* Conflicting records across pipeline stages
* Low-support BINs (below threshold) handled via exclusion or separate flag

---

#### **Testing Strategy**

* Unit tests:

  * Aggregation logic
  * Country code validation
  * Discrepancy detection
* Integration tests:

  * Monthly re-run simulation
  * Output schema and record count sanity checks
* Regression tests:

  * Compare outputs to previous runs to detect drift or anomalies

---

Next step: generate `prompt_plan.md`. Would you like me to begin breaking this into codegen units?
