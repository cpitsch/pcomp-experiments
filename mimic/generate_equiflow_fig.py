import pandas as pd
from equiflow import EquiFlow

data = pd.read_csv("results/equiflow.csv")

others = set()


def race_to_simpler_race(race: str) -> str:
    return race.split("-")[0].strip().split("/")[0].strip().title()


data["simpler_race"] = data["race"].apply(race_to_simpler_race)
top_k = data["simpler_race"].value_counts().nlargest(6).index


def convert_to_other(race: str) -> str:
    if race in top_k:
        return race
    else:
        others.add(race)
        print(f"Other {race}")
        return "Other"


# data["Race"] = data["simpler_race"].apply(lambda x: x if x in top_k else "Other")
data["Race"] = data["simpler_race"].apply(convert_to_other)

data = data.rename(
    columns={"gender": "Gender", "insurance": "Insurance", "anchor_age": "Age"}
)

eq = EquiFlow(
    data,
    initial_cohort_label="in MIMIC-IV",
    categorical=["Gender", "Insurance", "Race"],
    nonnormal=["Age"],
)

eq.add_exclusion(
    mask=data.has_gi_bleeding,
    exclusion_reason="no GI Bleeding",
    new_cohort_label="have GI Bleeding",
)

eq.add_exclusion(
    mask=(data.Age >= 18),
    exclusion_reason="being underaged",
    new_cohort_label="Final Cohort",
)

eq.plot_flows(smds=False, box_width=3)
print(others)
