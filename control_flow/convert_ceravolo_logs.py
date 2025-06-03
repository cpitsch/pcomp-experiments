from pathlib import Path

import pandas as pd
from pm4py import write_xes
from tqdm.auto import tqdm

# I have since replaced this script with a rust wrapper around the process_mining
# crate. But it does the same thing.
# `csv2xes -i ceravolo_raw -o testing_logs/ceravolo --case-id case --activity event --timestamp completeTime`

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


def convert_noiseless_logs():
    convert_logs(Path("ceravolo_raw"), Path("testing_logs", "ceravolo"))


def convert_noisy_logs():
    convert_logs(Path("ceravolo_raw_noisy"), Path("testing_logs_noisy", "ceravolo"))


def convert_logs(in_path: Path, out_path: Path):
    paths = list(in_path.rglob("*.csv"))
    for csv_path in tqdm(paths, desc="Converting CSV Files", unit="files"):
        df = pd.read_csv(
            csv_path,
            parse_dates=[INPUT_START_TIMESTAMP_COLUMN, INPUT_COMPLETE_TIMESTAMP_COLUMN],
        )
        log = df.rename(columns=COLUMN_MAPPING)
        log["case:concept:name"] = log["case:concept:name"].astype(str)

        log_path = out_path / (csv_path.with_suffix(".xes.gz").relative_to(in_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            # print("Writing", log_path.as_posix())
            write_xes(log, log_path.as_posix(), show_progress_bar=False)


if __name__ == "__main__":
    # convert_noiseless_logs()
    convert_noisy_logs()
