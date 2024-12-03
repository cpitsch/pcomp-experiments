from pathlib import Path

import numpy as np
import pandas as pd
from pcomp.utils.constants import (
    DEFAULT_NAME_KEY,
    DEFAULT_START_TIMESTAMP_KEY,
    DEFAULT_TIMESTAMP_KEY,
    DEFAULT_TRACEID_KEY,
)
from pm4py import write_xes

DEFAULT_RESOURCE_KEY = "org:resource"


CASE_ID_COLUMN = "stay_id"  # The column to use as a case notion


LOG_OUTPUT_PATH = Path("gi_bleeding.xes.gz")
QUERY_RESULTS_PATH = Path("results")


def prepare_transfusions_data(path: Path) -> pd.DataFrame:
    TRANSFUSIONS_COLUMN_MAPPING = {
        "starttime": DEFAULT_START_TIMESTAMP_KEY,
        "endtime": DEFAULT_TIMESTAMP_KEY,
        "item_label": DEFAULT_NAME_KEY,
        "caregiver_id": DEFAULT_RESOURCE_KEY,
    }

    transfusions = pd.read_csv(path, parse_dates=["starttime", "endtime"]).rename(
        columns=TRANSFUSIONS_COLUMN_MAPPING
    )
    transfusions[DEFAULT_TRACEID_KEY] = transfusions[CASE_ID_COLUMN].astype(str)

    return transfusions


def prepare_hemoglobin_measurement_data(path: Path) -> pd.DataFrame:
    HEMOGLOBIN_COLUMN_MAPPING = {
        "label": DEFAULT_NAME_KEY,
        "charttime": DEFAULT_START_TIMESTAMP_KEY,
        "storetime": DEFAULT_TIMESTAMP_KEY,
    }
    hemoglobin_measurements = pd.read_csv(
        path,
        parse_dates=["charttime", "storetime"],
        dtype={"ref_range_lower": np.float64, "ref_range_upper": np.float64},
    ).rename(columns=HEMOGLOBIN_COLUMN_MAPPING, errors="raise")

    hemoglobin_measurements[DEFAULT_TRACEID_KEY] = hemoglobin_measurements[
        CASE_ID_COLUMN
    ].astype(str)
    hemoglobin_measurements["hemoglobin_value"] = hemoglobin_measurements["valuenum"]
    hemoglobin_measurements: pd.DataFrame = hemoglobin_measurements[  # type: ignore
        ~hemoglobin_measurements["hemoglobin_value"].isna()
    ]
    hemoglobin_measurements["is_low_hemoglobin"] = (
        hemoglobin_measurements["hemoglobin_value"] < 7
    )

    hemoglobin_measurements.loc[
        (hemoglobin_measurements["hemoglobin_value"] < 7), "concept:name"
    ] = "Hemoglobin (Low)"
    hemoglobin_measurements.loc[
        (hemoglobin_measurements["hemoglobin_value"] >= 7), "concept:name"
    ] = "Hemoglobin (Normal)"

    return hemoglobin_measurements


# def prepare_outcome_data(path: Path) -> pd.DataFrame:
#     outcome_info = pd.read_csv(path, parse_dates=["admittime", "dischtime"])
#
#     ADMISSION_COLUMN_MAPPING = {"admittime": DEFAULT_TIMESTAMP_KEY}
#     # admissions: pd.DataFrame = outcome_info[
#     #     ["subject_id", "hadm_id", "stay_id", "admittime"]
#     # ].copy()  # type: ignore
#     admissions = outcome_info.drop(
#         columns=["dischtime", "hospital_expire_flag", "subject_id_1", "hadm_id_1"]
#     ).rename(columns=ADMISSION_COLUMN_MAPPING, errors="raise")
#
#     admissions[DEFAULT_NAME_KEY] = "ADMIT"
#     admissions[DEFAULT_START_TIMESTAMP_KEY] = admissions[DEFAULT_TIMESTAMP_KEY]
#     admissions[DEFAULT_TRACEID_KEY] = admissions[CASE_ID_COLUMN].astype(str)
#
#     DISCHARGE_COLUMN_MAPPING = {"dischtime": DEFAULT_TIMESTAMP_KEY}
#     # discharges = outcome_info[
#     #     ["subject_id", "hadm_id", "stay_id", "dischtime", "hospital_expire_flag"]
#     # ].rename(
#     #     columns=DISCHARGE_COLUMN_MAPPING, errors="raise"
#     # )  # type: ignore
#     discharges = outcome_info.drop(
#         columns=[
#             "admittime",
#             "subject_id_1",
#             "hadm_id_1",
#         ]
#     ).rename(columns=DISCHARGE_COLUMN_MAPPING, errors="raise")
#
#     discharges[DEFAULT_START_TIMESTAMP_KEY] = discharges[DEFAULT_TIMESTAMP_KEY]
#     discharges[DEFAULT_NAME_KEY] = discharges["hospital_expire_flag"].apply(
#         lambda died_in_hosp: "DEATH" if died_in_hosp else "DISCHARGE"
#     )
#     discharges[DEFAULT_TRACEID_KEY] = discharges[CASE_ID_COLUMN].astype(str)
#
#     return (
#         pd.concat([admissions, discharges])
#         .drop(columns=["hospital_expire_flag"])
#         .sort_values(by=[DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY])
#     )


def prepare_case_endpoints(path: Path) -> pd.DataFrame:
    # subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime,
    # los, gender, anchor_age, anchor_year, anchor_year_group, insurance, language,
    # marital_status, race, deathtime, icu_expire_flag, hospital_expire_flag

    df = pd.read_csv(path, parse_dates=["intime", "outtime"]).drop(
        columns=[
            "first_careunit",
            "last_careunit",
            "los",
            "deathtime",
            "hospital_expire_flag",
        ]
    )
    print(df["icu_expire_flag"].value_counts())

    df[DEFAULT_TRACEID_KEY] = df[CASE_ID_COLUMN].astype(str)

    START_EVENT_COLUMN_MAPPING = {"intime": DEFAULT_TIMESTAMP_KEY}
    start_events = df.drop(columns=["icu_expire_flag", "outtime"]).rename(
        columns=START_EVENT_COLUMN_MAPPING, errors="raise"
    )
    start_events[DEFAULT_NAME_KEY] = "ICU_TRANSFER"
    start_events[DEFAULT_START_TIMESTAMP_KEY] = start_events[
        DEFAULT_TIMESTAMP_KEY
    ]  # Atomic Event

    END_EVENTS_COLUMN_MAPPING = {"outtime": DEFAULT_TIMESTAMP_KEY}
    end_events = df.drop(columns=["intime"]).rename(columns=END_EVENTS_COLUMN_MAPPING)

    end_events[DEFAULT_START_TIMESTAMP_KEY] = end_events[DEFAULT_TIMESTAMP_KEY]
    end_events[DEFAULT_NAME_KEY] = end_events["icu_expire_flag"].apply(
        lambda died_in_icu: "DEATH" if died_in_icu else "ICU_DISCHARGE"
    )
    end_events = end_events.drop(columns=["icu_expire_flag"])

    return pd.concat([start_events, end_events]).sort_values(
        by=[DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY]
    )


def add_time_since_last_measurement(log: pd.DataFrame) -> pd.DataFrame:
    """Add "time_since_last_measurement" and "time_of_last_measurement" columns
    For events with no preceding hemoglobin measurement, this is 0. Otherwise, even for
    hemoglobin measurement events, it is the time since the preceding hemoglobin measurement
    event (event_type == hemoglobin_measurement)

    Args:
        log (pd.DataFrame): The event log.

    Returns:
        pd.DataFrame: The event log with the new columns
    """
    # aux_marker is used to mark events that are auxilliary, to be able to filter them out later
    log["aux_marker"] = 0
    aux_events = log[log["event_type"] == "hemoglobin_measurement"].copy()
    aux_events["aux_marker"] = 1
    aux_events["time_of_last_measurement"] = aux_events["time:timestamp"]
    log = pd.concat([log, aux_events]).sort_values(  # type: ignore
        by=["case:concept:name", "time:timestamp", "aux_marker"], ascending=True
    )
    # Disable sorting because the event log is already sorted and it can mess up timestamp ordering
    log["time_of_last_measurement"] = log.groupby("case:concept:name", sort=False)[
        "time_of_last_measurement"
    ].ffill()
    log["time_of_last_measurement"] = log["time_of_last_measurement"].fillna(
        value=log["time:timestamp"]
    )
    log["time_since_last_measurement"] = (
        log["time:timestamp"] - log["time_of_last_measurement"]
    ).dt.total_seconds()

    # Drop aux events
    log = log[log["aux_marker"] == 0].drop(columns=["aux_marker"])  # type: ignore

    return log


def main():
    transfusions_data = prepare_transfusions_data(
        QUERY_RESULTS_PATH / "blood_transfusions_gi_patients.csv"
    )
    hemoglobin_data = prepare_hemoglobin_measurement_data(
        QUERY_RESULTS_PATH / "hemoglobin_measurements_gi_patients.csv"
    )
    endpoints_data = prepare_case_endpoints(QUERY_RESULTS_PATH / "icu_outcomes.csv")

    ## Make a "comparison_value" column that holds the data of interest for each kind of event
    hemoglobin_data["comparison_value"] = hemoglobin_data["hemoglobin_value"]
    transfusions_data["comparison_value"] = transfusions_data["amount"]
    endpoints_data["comparison_value"] = 0.0

    ## Make the analysis easier by adding a column indicating the dataframe the row came from
    hemoglobin_data["event_type"] = "hemoglobin_measurement"
    transfusions_data["event_type"] = "blood_transfusion"
    endpoints_data["event_type"] = "endpoint"

    log = pd.concat([transfusions_data, hemoglobin_data, endpoints_data]).sort_values(
        by=[DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY], ascending=True
    )

    log = add_time_since_last_measurement(log)

    # Add new "time since last measurement" column
    # This has the time since the last measurement, which follows the following rules:
    # - 0 if the event itself is a measurement
    # - 0 if there was no preceding measurement
    # - The time since the last measurement, otherwise

    # log["time_of_last_measurement"] = log[DEFAULT_TIMESTAMP_KEY].where(
    #     log[DEFAULT_NAME_KEY] == "Hemoglobin"
    # )
    # log = log.sort_values([DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY])
    # log["time_of_last_measurement"] = log.groupby(DEFAULT_TRACEID_KEY, sort=False)[
    #     "time_of_last_measurement"
    # ].ffill()
    # log["time_of_last_measurement"] = log["time_of_last_measurement"].fillna(
    #     value=log[DEFAULT_TIMESTAMP_KEY]
    # )
    # log["time_since_last_measurement"] = (
    #     log[DEFAULT_TIMESTAMP_KEY] - log["time_of_last_measurement"]
    # ).dt.total_seconds()

    # Do the same for the "time since last transfusion" column
    # log["time_of_last_transfusion"] = log[DEFAULT_TIMESTAMP_KEY].where(
    #     log["event_type"] == "blood_transfusion"
    # )
    # log = log.sort_values([DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY])
    # log["time_of_last_transfusion"] = log.groupby(DEFAULT_TRACEID_KEY, sort=False)[
    #     "time_of_last_transfusion"
    # ].ffill()
    # log["time_of_last_transfusion"] = log["time_of_last_transfusion"].fillna(
    #     value=log[DEFAULT_TIMESTAMP_KEY]
    # )
    # log["time_since_last_transfusion"] = (
    #     log[DEFAULT_TIMESTAMP_KEY] - log["time_of_last_transfusion"]
    # ).dt.total_seconds()
    # log = log.drop(columns=["time_of_last_measurement", "time_of_last_transfusion"])

    write_xes(log, LOG_OUTPUT_PATH.as_posix())

    print("Log Summary:")
    print("\tSubjects:", log["subject_id"].nunique())
    print("\tHospital Admissions:", log["hadm_id"].nunique())
    print("\tICU Stays:", log["stay_id"].nunique())
    print("\tCases:", log[DEFAULT_TRACEID_KEY].nunique())
    print("\tEvents:", log.shape[0])


if __name__ == "__main__":
    main()
