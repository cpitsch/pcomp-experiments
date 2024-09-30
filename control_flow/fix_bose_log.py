"""
The bose log reuses case-ids for each sub-log region. As such, since the new pm4py 
version uses dataframes, these cases, when grouping the dataframe by case id, are merged,
which is undersired.

This script splits the event log into these sublogs based on time, then reassigns case ids
in order to guarantee their uniqueness.
"""

from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pm4py  # type: ignore

TIMESTAMP_COLUMN = "time:timestamp"
CASEID_COLUMN = "case:concept:name"

LOG_IN_PATH = "bose_log.xes.gz"
LOG_OUT_PATH = Path("testing_logs", "bose", "bose_log.xes.gz")

log = cast(pd.DataFrame, pm4py.read_xes(LOG_IN_PATH))
log[TIMESTAMP_COLUMN] = pd.to_datetime(log[TIMESTAMP_COLUMN])

# Fix the log
"""
Observation from Disco:
The log has 5 bunches of activity, spread over time. These bunches (very likely) correspond to the
4 drifts in the process.

"Time-Split-Points" taken from the Disco "Events Over Time" view:
- <zero>
- 1972-01-01 00:00:00
- 1975-05-05 00:00:00
- 1977-07-07 00:00:00
- 1979-09-09 00:00:00
- <today>
"""


zero_day = datetime.fromtimestamp(0)
today = datetime.now()

_split_points = [
    zero_day,
    datetime(1972, 1, 1),
    datetime(1975, 5, 5),
    datetime(1977, 7, 7),
    datetime(1979, 9, 9),
    today,
]

# '%Y-%m-%d %H:%M:%S'
split_points = [point.strftime(r"%Y-%m-%d %H:%M:%S") for point in _split_points]


logs = [
    # Keep only events in range since all traces will have an event in the range
    # Since for each case-id there is a separate case with that id in every range
    cast(pd.DataFrame, pm4py.filter_time_range(log, start, end, "events")).copy(
        deep=True
    )
    for start, end in zip(split_points, split_points[1:])
    if print("Filtering for range", start, "to", end) or True
]


CASEIDS_PER_LOG = logs[0][CASEID_COLUMN].nunique()

# Alter the caseids accordingly!
for idx, sub_log in enumerate(logs):
    base_caseid = idx * CASEIDS_PER_LOG
    sub_log[CASEID_COLUMN] = sub_log[CASEID_COLUMN].apply(
        # Assumes that the case_id column is integers formatted as string
        lambda case_id: str(base_caseid + int(case_id))
    )

    if idx > 0 and len(sub_log[sub_log[CASEID_COLUMN] == "2"]) > 0:
        raise Exception("Case-ID Reassignment didn't work")


# Stitch everything back together
mega_log = pd.concat(logs)
print("Number of cases in fixed log:", mega_log[CASEID_COLUMN].nunique())

# Just for convenience, order the Event Log by caseid
mega_log.sort_values(by=CASEID_COLUMN, key=pd.to_numeric, inplace=True)  # type: ignore


print("Log Preview:")
print(mega_log.head(40))

# Save the log as xes.gz
LOG_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pm4py.write_xes(mega_log, LOG_OUT_PATH.as_posix())
