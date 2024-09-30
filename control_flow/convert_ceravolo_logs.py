from pathlib import Path

import pandas as pd
from pm4py import write_xes

IN_PATH = Path("ceravolo_raw")


OUT_PATH = Path("testing_logs", "ceravolo")

INPUT_CASEID_COLUMN = "case"
INPUT_ACTIVITY_COLUMN = "event"
INPUT_COMPLETE_TIMESTAMP_COLUMN = "completeTime"
INPUT_START_TIMESTAMP_COLUMN = "startTime"

COLUMN_MAPPING = {
    INPUT_CASEID_COLUMN: "case:concept:name",
    INPUT_ACTIVITY_COLUMN: "concept:name",
    INPUT_COMPLETE_TIMESTAMP_COLUMN: "time:timestamp",
    INPUT_START_TIMESTAMP_COLUMN: "start_timestamp",
}


def main():
    for csv_path in IN_PATH.glob("*.csv"):
        df = pd.read_csv(
            csv_path,
            parse_dates=[INPUT_START_TIMESTAMP_COLUMN, INPUT_COMPLETE_TIMESTAMP_COLUMN],
        )
        log = df.rename(columns=COLUMN_MAPPING)
        log["case:concept:name"] = log["case:concept:name"].astype(str)

        log_path = OUT_PATH / (csv_path.with_suffix(".xes.gz").name)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        write_xes(log, log_path.as_posix())


if __name__ == "__main__":
    main()
