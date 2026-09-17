# Raw response locator

Every successful primary cell stores `raw_response` in:

- SQLite: `/Users/thomas/Desktop/trustOrCheckMe/results/study1_causal_pilot/study1.sqlite3` table `requests.record_json` and `attempts.raw_output`
- CSV: `primary_results.csv` column `raw_response`

Task-002 reused cells keep their original smoke-test records in the same Study-1 sqlite.
Historical V2 remains at `results/v2/raw/v2.sqlite3` and was opened read-only.
